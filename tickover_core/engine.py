"""Drift scoring: how far a clip sits from one machine's own healthy recordings.

The recipe follows the training-free systems at the top of DCASE Task 2 (2024-2026):
frozen pretrained encoders, a memory bank of healthy recordings, 1-nearest-neighbour
cosine distance, and an ensemble across two different encoders.

Three scores, each calibrated on the machine itself:

  A. CED band bank. CED-base (a ViT distilled on AudioSet) turns a clip into a grid
     of tokens over time x 4 frequency bands. Per band, the clip is pooled two ways,
     a plain mean and relative deviation pooling (RDP, Wilkinghoff et al. 2026),
     which up-weights frames that stray from the clip mean, so a single click or
     stall survives pooling. Score = 1 - best cosine to the bank, per band (BEAM,
     Saengthong et al. 2026), after centring on the bank mean.
  B. mn10 clip match. EfficientAT mn10 (a MobileNetV3, the Injini encoder): the
     clip's mean embedding, 1 - best cosine to the bank.
  C. mn10 frame match. Each frame's distance to its nearest healthy frame; score =
     the 90th percentile, so brief events count.

Calibration: every healthy clip is scored against the others (leave one out). Each
score becomes robust sigmas (median / MAD) against that spread; drift is the mean
of the three. No labels, no training. On DCASE 2024 robot, drone, motor, printer and
compressor recordings with 10 healthy clips, the ensemble beats each part alone
(scripts in bench/, numbers in the README).
"""
from __future__ import annotations

import time

import numpy as np

from . import audio, encoder, quality, store

GAMMA = 10.0
CED_LAYER = "L5"
MIN_CLIPS = 3
WATCH, DRIFT = 2.0, 3.0
CHECK_MIN_S = 10.0  # the baseline is built from 10-second clips; 9.9 s already scores as change
FLOOR = 0.15  # MAD floor as a fraction of the median: few-shot spreads are noisy
LEVEL_SHIFT_DB = 8.0


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _rdp(g: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    m = g.mean(0, keepdims=True)
    d = np.linalg.norm(g - m, axis=-1)
    w = (1 + d / (d.max(0, keepdims=True) + 1e-9)) ** gamma
    return (w[..., None] * g).sum(0) / w.sum(0)[..., None]


def _pools(g: np.ndarray) -> np.ndarray:
    """(T, B, D) token grid -> (2, B, D): mean and RDP per band."""
    return np.stack([g.mean(0), _rdp(g)])


# ---------------------------------------------------------------- the three scores
def score_ced(bank_pools: np.ndarray, q_pools: np.ndarray) -> np.ndarray:
    """-> (B,) per-band distance."""
    mu = bank_pools.mean(0)
    b = _l2(bank_pools - mu)
    q = _l2(q_pools - mu)
    return (1 - np.einsum("npbd,pbd->npb", b, q).max(0)).mean(0)


def score_mn_clip(bank_means: np.ndarray, q_frames: np.ndarray) -> float:
    return float(1 - (_l2(bank_means) @ _l2(q_frames.mean(0))).max())


def score_mn_frames(bank_frames: np.ndarray, q_frames: np.ndarray, q: float = 0.9) -> float:
    return float(np.quantile(1 - (_l2(q_frames) @ _l2(bank_frames).T).max(1), q))


def _robust(v: np.ndarray):
    v = np.asarray(v, np.float64)
    med = np.median(v, axis=0)
    mad = 1.4826 * np.median(np.abs(v - med), axis=0)
    return med, np.maximum(mad, FLOOR * np.abs(med) + 1e-6)


def _token_map(bank_tokens: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per (time, band) token: 1 - cosine to the nearest healthy token of that band. -> (B, T)."""
    mu = bank_tokens.mean(0)
    out = np.zeros((g.shape[1], g.shape[0]), np.float32)
    for b in range(g.shape[1]):
        out[b] = 1 - (_l2(g[:, b] - mu[b]) @ _l2(bank_tokens[:, b] - mu[b]).T).max(1)
    return out


# ---------------------------------------------------------------- features
def _features(x: np.ndarray) -> dict:
    p, g = encoder.get().run(x, CED_LAYER)
    g = g.astype(np.float32)
    f = encoder.get_mn10().frames(x).astype(np.float32)
    return {"grid": g, "pools": _pools(g), "frames": f, "probs": p}


def _usable(paths: list[str], ignore_quality: bool = False) -> tuple[list[dict], list[str]]:
    """Recordings -> features of every 10 s segment that passes the quality gate.

    A recording that is clipped, silent or too short is left out whole. A segment with speech or wind over it
    is left out alone, so one remark in a minute of healthy sound costs ten seconds. Each feature dict carries
    `src` (the index of its recording in `paths`) and `level` (dBFS).
    """
    F, skipped = [], []
    for gi, p in enumerate(paths):
        try:
            x = audio.load(p)
            segs = audio.segments(x)
        except ValueError as exc:
            skipped.append(f"{p}: {exc}")
            continue
        sig = quality.assess(x)
        if not sig["ok"] and not ignore_quality:
            skipped.append(f"{p}: {sig['fix']}")
            continue
        for seg in segs:
            f = _features(seg)
            why = None if ignore_quality else quality.segment_reason(f["probs"])
            if why:
                skipped.append(f"{p}: {quality.FIX[why]}")
                continue
            f["src"], f["level"] = gi, audio.level_db(seg)
            F.append(f)
    if not F:
        raise ValueError("no usable audio: " + "; ".join(skipped))
    return F, skipped


def _bank_scores(F: list[dict], keep: list[int], q: dict) -> np.ndarray:
    """Raw scores of query features q against bank members `keep` -> [ced_bands, mn_clip, mn_frames]."""
    P = np.stack([F[i]["pools"] for i in keep])
    band = score_ced(P, q["pools"])
    mc = score_mn_clip(np.stack([F[i]["frames"].mean(0) for i in keep]), q["frames"])
    mf = score_mn_frames(np.concatenate([F[i]["frames"] for i in keep]), q["frames"])
    return np.r_[band.mean(), mc, mf]


# ---------------------------------------------------------------- public API
def enroll(machine: str, paths: list[str], append: bool = False, ignore_quality: bool = False) -> dict:
    F, skipped = _usable(paths, ignore_quality)
    if append:
        try:
            old, _ = store.load_bank(machine)
            F = _unpack(old) + F
        except KeyError:
            pass
    n = len(F)
    meta = {"name": machine, "n_clips": n, "encoder": "ced-base + mn10", "enrolled_at": time.time(),
            "sources": [str(p) for p in paths]}
    if n < MIN_CLIPS:
        store.save_bank(machine, _pack(F, None), {**meta, "ready": False})
        return {"ok": True, "machine": machine, "ready": False, "n_clips": n, "skipped": skipped,
                "message": f"Baseline has {n} clip(s). Send {MIN_CLIPS - n} more healthy recording(s) of about 10 s, "
                           f"or one recording of 30 s or more, to finish it."}
    loo = np.stack([_bank_scores(F, [j for j in range(n) if j != i], F[i]) for i in range(n)])
    med, mad = _robust(loo)
    zl = ((loo - med) / mad)[:, :3].mean(1)
    store.save_bank(machine, _pack(F, (med, mad)), {**meta, "ready": True})
    return {"ok": True, "machine": machine, "ready": True, "n_clips": n, "skipped": skipped,
            "healthy_spread_sigma": round(float(np.max(np.abs(zl))), 2),
            "message": f"Enrolled {n} healthy clips. Tickover now knows how '{machine}' sounds."
                       + (f" Left out {len(skipped)} that failed the quality gate." if skipped else "")}


def _retake(machine: str, q: dict) -> dict:
    return {"ok": True, "machine": machine, "verdict": "retake", "quality": q, "logged": False,
            "message": "Not scored and not logged. " + q["fix"],
            "how_to_read": "retake is about the recording, never about the machine. Ask for a new recording; "
                           "do not report any verdict on the machine from this one."}


def score(machine: str, x: np.ndarray) -> dict:
    """Drift of a waveform against a machine's baseline. Nothing is logged and no card is drawn."""
    bank, meta = store.load_bank(machine)
    if not meta.get("ready"):
        return {"machine": meta["name"], "verdict": "insufficient"}
    F = _unpack(bank)
    zz = max(float(((_bank_scores(F, list(range(len(F))), _features(s)) - bank["cal_med"]) / bank["cal_mad"])[:3].mean())
             for s in audio.segments(x))
    return {"machine": meta["name"], "drift_sigma": round(zz, 2),
            "verdict": "drift" if zz >= DRIFT else "watch" if zz >= WATCH else "healthy"}


def check(machine: str, path: str, out_png: str | None = None, recorded_at: float | None = None,
          ignore_quality: bool = False) -> dict:
    bank, meta = store.load_bank(machine)
    if not meta.get("ready"):
        return {"ok": True, "machine": machine, "verdict": "insufficient", "n_clips": meta["n_clips"],
                "message": f"Baseline has {meta['n_clips']} clip(s); it needs {MIN_CLIPS}. Enroll more healthy sound first."}
    F = _unpack(bank)
    med, mad = bank["cal_med"], bank["cal_mad"]
    x = audio.load(path)
    if len(x) < 2 * audio.SR:
        return _retake(meta["name"], quality.assess(x))
    if len(x) < CHECK_MIN_S * audio.SR:   # shorter clips score as change against a 10-second baseline
        q = quality.assess(x)
        q = {**q, "ok": False, "fix": f"This recording is {len(x) / audio.SR:.1f} s. Tickover compares 10-second clips. "
                                      "Record at least 10 seconds."}
        return _retake(meta["name"], q)
    segs = audio.segments(x)
    Q = [_features(seg) for seg in segs]
    ql = quality.assess(x, np.stack([q["probs"] for q in Q]))
    if not ql["ok"] and not ignore_quality:
        return _retake(meta["name"], ql)
    best = None
    for seg, q in zip(segs, Q):
        z = (_bank_scores(F, list(range(len(F))), q) - med) / mad
        zz = float(z[:3].mean())
        if best is None or zz > best[0]:
            best = (zz, z, q, seg)
    zz, z, q, seg = best
    verdict = "drift" if zz >= DRIFT else "watch" if zz >= WATCH else "healthy"
    # Where it changed: token-level distances localise frequency (pooled band scores do not,
    # attention mixes bands). Scale: 2.0 = the healthy 95th percentile, like a z of 2.
    tok = _token_map(np.concatenate([f["grid"] for f in F]), q["grid"])
    ref = bank["tok_ref"]
    amap = (tok - ref[0][:, None]) / (ref[1][:, None] - ref[0][:, None] + 1e-6) * 2.0
    band_z = np.quantile(amap, 0.9, axis=1)
    past = store.last_checks(machine, 29)
    # The same audio checked twice is one observation: it is not logged again and cannot confirm itself.
    import hashlib
    sig = f"{path}#{hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]}"
    repeat = bool(past and past[0].get("src") == sig)
    hist = [h["z"] for h in reversed(past)] + ([] if repeat else [zz])
    png = out_png or str(store.machine_dir(machine) / f"check_{int(time.time() * 1000)}.png")
    from . import card
    card.render(png, machine=meta["name"], verdict=verdict, z=zz, mel=_logmel(seg), amap=np.clip(amap, 0, None),
                band_labels=encoder.BAND_LABELS, band_z=[float(v) for v in band_z], history=hist,
                n_enrolled=meta["n_clips"], enrolled_on=time.strftime("%d %b %Y", time.localtime(meta["enrolled_at"])),
                encoder="CED-base + mn10")
    if not repeat:
        store.log_check(machine, float(z[:3].mean()), zz, verdict, png, sig, ts=recorded_at)
    from . import trend as _trend
    prof = store.profile(machine)
    tr = _trend.analyse(store.history(machine), DRIFT, WATCH, every_days=prof.get("every_days"))
    top = int(np.argmax(band_z))
    out = {"ok": True, "machine": meta["name"], "verdict": verdict, "drift_sigma": round(zz, 2), "trend": tr,
           "scores_sigma": {"ced_bands": round(float(z[0]), 2), "mn10_clip": round(float(z[1]), 2),
                            "mn10_frames": round(float(z[2]), 2)},
           "bands": {lab: round(float(v), 2) for lab, v in zip(encoder.BAND_LABELS, band_z)},
           "most_changed_band": encoder.BAND_LABELS[top] if band_z[top] >= WATCH else None,
           "segments_scored": len(segs), "quality": ql, "card": png,
           "how_to_read": "drift_sigma is robust sigmas from this machine's own healthy recordings. "
                          "Under 2 is normal variation, 2-3 is worth watching, 3 and above is a real change. "
                          "Tickover detects change. It does not name the fault."}
    # Recording conditions: a clip far louder or quieter than every baseline clip was probably recorded from a
    # different distance, which alone can read as drift. Advisory; banks from before 0.3 have no levels.
    lv = bank["levels"] if "levels" in bank else None
    if lv is not None and np.isfinite(lv).any():
        lv = lv[np.isfinite(lv)]
        now = audio.level_db(seg)
        d = now - float(np.median(lv))
        if abs(d) > LEVEL_SHIFT_DB and not (lv.min() - 3 <= now <= lv.max() + 3):
            out["conditions"] = (f"This recording is {abs(d):.0f} dB {'louder' if d > 0 else 'quieter'} than the baseline "
                                 "recordings. The phone was probably at a different distance. Ask for one more from the "
                                 "usual spot before acting on this verdict.")
    if repeat:
        out["repeat"] = True
        out["note"] = "This is the recording that was checked last time. It was not logged again."
    if verdict == "drift":
        earlier = past[1:] if repeat else past
        out["confirmed"] = bool(earlier and earlier[0]["verdict"] == "drift")
        if not out["confirmed"]:
            out["confirm"] = ("First drift reading. Ask for a second recording now, from the same spot, and check it "
                              "before telling the owner to act. Two in a row is a finding; one can be a bad recording.")
    return out


# ---------------------------------------------------------------- storage packing
def _pack(F: list[dict], cal) -> dict:
    lens_g = np.array([f["grid"].shape[0] for f in F])
    lens_f = np.array([f["frames"].shape[0] for f in F])
    out = {"grids": np.concatenate([f["grid"] for f in F]).astype(np.float16), "lens_g": lens_g,
           "frames": np.concatenate([f["frames"] for f in F]).astype(np.float16), "lens_f": lens_f,
           "pools": np.stack([f["pools"] for f in F]).astype(np.float32),
           "levels": np.array([f.get("level", np.nan) for f in F], np.float32)}
    if cal is not None:
        out["cal_med"], out["cal_mad"] = cal
        # healthy token-distance reference for the card's colour scale (leave one out)
        maps = []
        for i in range(len(F)):
            others = np.concatenate([F[j]["grid"] for j in range(len(F)) if j != i])
            maps.append(_token_map(others, F[i]["grid"]))
        allm = np.concatenate(maps, axis=1)
        out["tok_ref"] = np.quantile(allm, [0.5, 0.95], axis=1)
    return out


def _unpack(bank: dict) -> list[dict]:
    G = np.split(bank["grids"].astype(np.float32), np.cumsum(bank["lens_g"])[:-1])
    Fr = np.split(bank["frames"].astype(np.float32), np.cumsum(bank["lens_f"])[:-1])
    lv = bank["levels"] if "levels" in bank else np.full(len(G), np.nan, np.float32)
    return [{"grid": g, "frames": f, "pools": p, "level": float(v)} for g, f, p, v in zip(G, Fr, bank["pools"], lv)]


def _logmel(x: np.ndarray) -> np.ndarray:
    n_fft, hop = 1024, 256
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (n_fft // 2, n_fft // 2)), n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    f = np.linspace(0, 8000, spec.shape[1])
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(0, mel(8000), 66) / 2595) - 1)
    fb = np.maximum(0, np.minimum((f[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None] + 1e-9),
                                  (edges[2:, None] - f[None]) / (edges[2:, None] - edges[1:-1, None] + 1e-9)))
    return np.log(spec @ fb.T + 1e-10)

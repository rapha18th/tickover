"""Verify: did the sound change when the action ran?

An agent that switches a pump on gets back "command sent". That says nothing about the pump.
Verify listens before and after, and answers from the sound.

  before / after   Two windows around an action. `before` stores a window. The agent then
                   does whatever it does (a smart plug, a relay, a shell command, a person
                   flipping a switch). `after` takes a second window and compares.
  clip             One recording with the change somewhere inside it. Verify finds the
                   moment that splits it best and compares the two sides.

The comparison needs no baseline and no training. It reads two things, each against the
windows' own variation, in robust sigmas:

  level   The loudness of every third of a second. The step between the two windows'
          medians, over the larger of their spreads. A machine starting next to a
          microphone is louder; this is the strong signal.
  tone    EfficientAT mn10 frames. Each window's frames, scored against that window's other
          frames (neighbours in time left out), give its own variation. The other window's
          frames are scored on that scale. The median counts, so more than half a window
          has to move and one door slam does not. Both directions are scored; the larger
          counts, because a steady window judges a noisy one better than the reverse.

The sound changed when level or tone crosses its line. The lines come from
bench/verify_bench.py: recordings split in half must not read as changed.

`expect` says what the action should have done: start (changed and louder), stop (changed
and quieter) or change. With a `machine`, the after window is also scored against that
machine's healthy baseline, so "it started" comes with "and it sounds like itself".
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import audio, encoder, engine, quality, store

TONE = 5.0            # tone sigma line, two separate windows
LEVEL = 4.0           # level sigma line
LEVEL_DB = 2.0        # and the step must be at least this many dB
TONE_CLIP = 8.0       # one clip: the split point is searched, so the lines are higher
LEVEL_CLIP = 5.0
LEVEL_BACKED = 3.0    # level sigma that counts for direction when the tone also changed
LEVEL_FLOOR = 0.5     # dB: a floor on a window's level spread
MIN_SIDE = 4.0        # seconds on each side: fewer frames than this and the encoder's edges dominate
GUARD = 2             # frames either side left out when a frame is scored against its own window
FRAME = 0.32          # seconds per level frame, about one mn10 frame
BEFORE_TTL = 3600
MAX_SECONDS = 120.0


EDGE = (2, 3)         # mn10 frames at the start and end of each 10 s pass are padding artefacts
FSEC = 0.3125         # seconds per mn10 frame (32 frames per 10 s)


def _frames_at(x: np.ndarray) -> tuple[np.ndarray, int]:
    """(mn10 frames, index of the first one) for a window of any length.

    The encoder takes 10 s at a time and its first and last frames are padding artefacts.
    Left in, every 10 s boundary and every short tail looks like a change. So passes overlap
    by half, each frame comes from the pass where it sits nearest the middle, and the frames
    at the very ends of the window are dropped.
    """
    mn = encoder.get_mn10()
    if len(x) < audio.SR:
        raise ValueError(f"window is {len(x) / audio.SR:.1f} s; verify needs at least {MIN_SIDE:.0f} s on each side")
    per = FSEC * audio.SR
    if len(x) <= encoder.CLIP:
        f = mn.frames(x)
        lo, hi = EDGE[0], len(f) - EDGE[1]
        if hi - lo < 2 * GUARD + 4:
            lo, hi = 0, len(f)
        f = f[lo:hi]
    else:
        starts = list(range(0, len(x) - encoder.CLIP + 1, encoder.CLIP // 2))
        if starts[-1] != len(x) - encoder.CLIP:
            starts.append(len(x) - encoder.CLIP)
        best: dict[int, tuple[float, np.ndarray]] = {}
        for st in starts:
            g = mn.frames(x[st:st + encoder.CLIP])
            base = int(round(st / per))
            for j in range(EDGE[0], len(g) - EDGE[1]):
                c = abs(j - (len(g) - 1) / 2)
                if base + j not in best or c < best[base + j][0]:
                    best[base + j] = (c, g[j])
        idx = sorted(best)
        lo, f = idx[0], np.stack([best[i][1] for i in idx])
    f = f.astype(np.float32)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9), lo


def frames(x: np.ndarray) -> np.ndarray:
    return _frames_at(x)[0]


def levels(x: np.ndarray) -> np.ndarray:
    """dB level of each FRAME of the window."""
    n = int(FRAME * audio.SR)
    fr = x[: len(x) // n * n].reshape(-1, n).astype(np.float64)
    fr = fr - fr.mean(1, keepdims=True)  # a DC offset is not sound
    return np.maximum(10 * np.log10((fr ** 2).mean(1) + 1e-12), LEVEL_MIN)


LEVEL_MIN = -75.0   # dBFS floor for frame levels: below this is the recorder's own noise
WHOLE_DB = 6.0      # a whole-window level difference this large gives the direction on its own


def _one_way(D: np.ndarray, a: slice, b: slice) -> float:
    """Median sigma of the frames in b against the frames in a, on a's own scale."""
    A = D[a, a].copy()
    i = np.arange(len(A))
    A[np.abs(i[:, None] - i[None, :]) <= GUARD] = np.inf
    med, mad = engine._robust(A.min(1))
    return float(np.median((D[b, a].min(1) - med) / mad))


def _tone(D: np.ndarray, t: int) -> float:
    a, b = slice(0, t), slice(t, len(D))
    return max(_one_way(D, a, b), _one_way(D, b, a))


def _level(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """(step in dB, step in sigmas of the noisier window)."""
    ma, mb = float(np.median(la)), float(np.median(lb))
    spread = max(1.4826 * float(np.median(np.abs(la - ma))), 1.4826 * float(np.median(np.abs(lb - mb))), LEVEL_FLOOR)
    return mb - ma, abs(mb - ma) / spread


def measure(xa: np.ndarray, xb: np.ndarray) -> dict:
    """Tone and level change between two windows."""
    fa, fb = frames(xa), frames(xb)
    F = np.concatenate([fa, fb])
    db, ls = _level(levels(xa), levels(xb))
    return {"tone_sigma": _tone(1 - F @ F.T, len(fa)), "level_delta_db": db, "level_sigma": ls}


def search(x: np.ndarray) -> tuple[int, dict]:
    """One clip: the sample that splits it into the two most different sides, and the change there."""
    (f, off), L = _frames_at(x), levels(x)
    n, secs = len(f), len(x) / audio.SR
    side = max(2 * GUARD + 4, int(np.ceil(MIN_SIDE / FSEC)) - sum(EDGE) // 2)
    if n < 2 * side or len(L) < 4:
        raise ValueError(f"clip is {secs:.0f} s; a one-clip verify needs at least {2 * MIN_SIDE + 1:.0f} s, "
                         "with a few seconds before the change and a few after")
    D = 1 - f @ f.T
    best = None
    for t in range(side, n - side + 1):
        k = min(max(int(round((t + off) * FSEC / FRAME)), 2), len(L) - 2)
        db, ls = _level(L[:k], L[k:])
        m = {"tone_sigma": _tone(D, t), "level_delta_db": db, "level_sigma": ls}
        key = max(m["tone_sigma"] / TONE_CLIP, ls / LEVEL_CLIP if abs(db) >= LEVEL_DB else 0.0)
        if best is None or key > best[0]:
            best = (key, t, m)
    sec, m = (best[1] + off) * FSEC, best[2]
    if abs(m["level_delta_db"]) >= LEVEL_DB:
        # Encoder frames blur a second or so. The level track is sharper: move the cut to the level step nearby.
        k0, w = int(round(sec / FRAME)), int(round(2.0 / FRAME))
        ks = range(max(2, k0 - w), min(len(L) - 2, k0 + w) + 1)
        k = min(ks, key=lambda i: float(((L[:i] - L[:i].mean()) ** 2).sum() + ((L[i:] - L[i:].mean()) ** 2).sum()))
        db, ls = _level(L[:k], L[k:])
        if abs(db) >= abs(m["level_delta_db"]):
            sec, m = k * FRAME, {**m, "level_delta_db": db, "level_sigma": ls}
    cut = int(round(sec * audio.SR))
    return min(max(cut, audio.SR), len(x) - audio.SR), m


def decide(m: dict, clip: bool = False) -> tuple[bool, str]:
    """(changed, direction) for a measurement."""
    tone_line, level_line = (TONE_CLIP, LEVEL_CLIP) if clip else (TONE, LEVEL)
    toned = m["tone_sigma"] >= tone_line
    big = abs(m["level_delta_db"]) >= LEVEL_DB
    stepped = big and m["level_sigma"] >= level_line
    # a tone change backs a smaller level step: together they give the direction
    directed = stepped or (big and toned and m["level_sigma"] >= LEVEL_BACKED)
    direction = "same level" if not directed else "louder" if m["level_delta_db"] > 0 else "quieter"
    return bool(stepped or toned), direction


def _tags(x: np.ndarray) -> np.ndarray:
    enc = encoder.get()
    return np.mean([enc.probs(x[s:s + encoder.CLIP]) for s in range(0, len(x), encoder.CLIP) if len(x) - s >= audio.SR], axis=0)


def _judge(xa: np.ndarray, xb: np.ndarray, m: dict, clip: bool, expect: str, label: str | None,
           machine: str | None, extra: dict) -> dict:
    names = encoder.labels()
    tone_line, level_line = (TONE_CLIP, LEVEL_CLIP) if clip else (TONE, LEVEL)
    changed, direction = decide(m, clip)
    la, lb = audio.level_db(xa - xa.mean()), audio.level_db(xb - xb.mean())
    silent = max(la, lb) < quality.TOO_QUIET_DBFS
    if silent:  # two windows of near silence differ only in noise nobody can hear
        changed, direction = False, "same level"
    elif changed and direction == "same level" and abs(lb - la) >= WHOLE_DB:
        # the change landed late in a window: the median step misses it, the whole-window level does not
        direction = "louder" if lb > la else "quieter"
        m = {**m, "level_delta_db": lb - la}
    tone, dlev, lsig = m["tone_sigma"], m["level_delta_db"], m["level_sigma"]
    pa, pb = _tags(xa), _tags(xb)
    diff = pb - pa
    appeared = [names[i] for i in np.argsort(-diff)[:3] if diff[i] >= 0.08 and names[i] != "Silence"]
    faded = [names[i] for i in np.argsort(diff)[:3] if diff[i] <= -0.08 and names[i] != "Silence"]
    want = {"start": "louder", "stop": "quieter"}.get(expect)
    how = (f"{abs(dlev):.0f} dB {direction}" if direction != "same level" else "the same level") + \
          (", with a different tone" if tone >= tone_line else "")
    if silent:
        verdict = "no_change"
        msg = (f"Both windows are near silence ({la:.0f} and {lb:.0f} dBFS). The microphone hears nothing to judge. "
               "Move it closer to the machine.")
    elif not changed:
        verdict = "no_change"
        msg = (f"No change heard. Level moved {dlev:+.1f} dB and tone {tone:+.1f} sigma, both inside the windows' own variation.")
    elif want and direction != want:
        verdict = "other_change"
        msg = (f"The sound changed, but it is {how}. A {expect} should be {want}. "
               "Something happened; it does not sound like the expected action.")
    else:
        verdict = "confirmed"
        msg = f"Confirmed by sound: it is {how} than before."  if direction != "same level" else \
              f"Confirmed by sound: the tone changed ({tone:+.1f} sigma) at the same level."
    out = {"ok": True, "verdict": verdict, "changed": changed, "expect": expect, "direction": direction,
           "level_delta_db": round(dlev, 1), "level_sigma": round(lsig, 1), "level_line_sigma": level_line,
           "tone_sigma": round(tone, 1), "tone_line_sigma": tone_line,
           "level_before_dbfs": round(la, 1), "level_after_dbfs": round(lb, 1),
           "appeared": appeared, "faded": faded,
           "seconds_before": round(len(xa) / audio.SR, 1), "seconds_after": round(len(xb) / audio.SR, 1), **extra}
    cautions = []
    if max(quality.signal(xa)["clipping_pct"], quality.signal(xb)["clipping_pct"]) > quality.CLIP_PCT:
        cautions.append("The recording is distorted; move the microphone back.")
    if max(pa[quality.I_SPEECH], pb[quality.I_SPEECH]) > quality.SPEECH:
        cautions.append("Someone is talking in one of the windows. Speech alone can read as a change; repeat without talking.")
    if cautions:
        out["cautions"] = cautions
    base = None
    if machine and verdict == "confirmed" and expect != "stop":
        try:
            sc = engine.score(machine, xb)
            out["baseline"] = sc
            if "drift_sigma" in sc:
                base = f"{sc['verdict']} ({sc['drift_sigma']:+.1f} σ)"
                msg += (f" Against {sc['machine']}'s healthy baseline the after window reads {sc['verdict']} "
                        f"({sc['drift_sigma']:+.1f} sigma). Not logged; a start-up is not a weekly check.")
        except (KeyError, ValueError) as exc:
            out["baseline"] = {"error": str(exc)}
    out["message"] = msg
    d = store.data_dir() / "verify"
    d.mkdir(exist_ok=True)
    png = str(d / f"verify_{int(time.time() * 1000)}.png")
    from . import card
    card.verify_card(png, label=label or {"start": "Did it start?", "stop": "Did it stop?"}.get(expect, "Did it change?"),
                     verdict=verdict, line=msg.split(". ")[0].rstrip(".") + ".", mel_before=engine._logmel(xa[: 30 * audio.SR]),
                     mel_after=engine._logmel(xb[: 30 * audio.SR]), sec_before=len(xa) / audio.SR, sec_after=len(xb) / audio.SR,
                     tone=tone, tone_line=tone_line, level_delta=dlev, level_sigma=lsig, level_line=level_line,
                     appeared=appeared, faded=faded, baseline=base)
    out["card"] = png
    out["how_to_read"] = ("level_sigma and tone_sigma are how far the after window sits from the before window, in units of "
                          "the windows' own variation. confirmed means the sound changed in the expected direction. "
                          "no_change means the action did not reach the machine, or the microphone cannot hear it. "
                          "Verify hears that something changed. It does not prove what changed.")
    return out


def _expect(v) -> str:
    v = str(v or "change").lower()
    if v not in ("start", "stop", "change"):
        raise ValueError("expect must be start, stop or change")
    return v


def _state():
    d = store.data_dir() / "verify"
    d.mkdir(exist_ok=True)
    return d / "before.json"


def _window(source, seconds, device) -> np.ndarray:
    x = audio.capture(source, seconds, device)[: int(MAX_SECONDS * audio.SR)]
    if len(x) < MIN_SIDE * audio.SR:
        raise ValueError(f"window is {len(x) / audio.SR:.1f} s; it needs at least {MIN_SIDE:.0f} s")
    return x


def before(source: str | None = None, seconds: float = 6.0, label: str | None = None, device=None) -> dict:
    """Store the window before an action."""
    import soundfile as sf
    x = _window(source, seconds, device)
    wav = _state().parent / "before.wav"
    sf.write(wav, x, audio.SR, subtype="FLOAT")
    _state().write_text(json.dumps({"wav": str(wav), "ts": time.time(), "label": label, "source": source}))
    return {"ok": True, "stored": "before", "seconds": round(len(x) / audio.SR, 1), "level_dbfs": round(audio.level_db(x), 1),
            "next": "Run the action now. Give the machine a few seconds to settle, then call tickover_verify "
                    "with step='after', the same source, and expect='start', 'stop' or 'change'."}


def after(source: str | None = None, seconds: float = 6.0, expect: str = "change", label: str | None = None,
          machine: str | None = None, before_path: str | None = None, device=None) -> dict:
    """Take the window after an action and compare it with the stored (or given) before window."""
    expect = _expect(expect)
    if before_path:
        xa, st = audio.load(before_path, max_seconds=MAX_SECONDS), {}
    else:
        if not _state().exists():
            raise ValueError("no before window. Call tickover_verify with step='before' first, or pass before_path.")
        st = json.loads(_state().read_text())
        if time.time() - st["ts"] > BEFORE_TTL:
            raise ValueError("the stored before window is over an hour old. Take a new one with step='before'.")
        xa = audio.load(st["wav"])
    if len(xa) < MIN_SIDE * audio.SR:
        raise ValueError(f"the before window needs at least {MIN_SIDE:.0f} s of sound")
    xb = _window(source, seconds, device)
    out = _judge(xa, xb, measure(xa, xb), False, expect, label or st.get("label"), machine, {"mode": "before_after"})
    if not before_path:
        _state().unlink(missing_ok=True)
    return out


def clip(path: str, expect: str = "change", label: str | None = None, machine: str | None = None) -> dict:
    """One recording with the action inside it: find the moment and compare the two sides."""
    expect = _expect(expect)
    if audio.is_url(path):
        path = audio.bridge_clip_url(path, 30)   # a bridge /clip records 10 s unless asked for more
    x = audio.load(path, max_seconds=MAX_SECONDS, url_seconds=30.0)
    cut, m = search(x)
    out = _judge(x[:cut], x[cut:], m, True, expect, label, machine,
                 {"mode": "clip", "change_at_s": round(cut / audio.SR, 1)})
    if not out["changed"]:
        out.pop("change_at_s", None)  # nothing changed, so there is no moment to report
    edge = edge_step(x)
    if out["direction"] == "same level" and edge >= WHOLE_DB:
        # The search cannot place a split within MIN_SIDE of either end, so a change there is not judged.
        out["verdict"] = "unclear"
        out.pop("change_at_s", None)
        out.pop("card", None)  # the card was drawn for a different verdict
        out["message"] = (f"The level moves about {edge:.0f} dB within {MIN_SIDE:.0f} s of the start or end of this "
                          f"recording. That is too close to the edge to judge. Record at least {MIN_SIDE:.0f} s before "
                          "the action and as long after it, then verify again.")
    return out


def edge_step(x: np.ndarray) -> float:
    """Largest level difference between either end of a clip and the rest of it, in dB."""
    L = levels(x)
    k = int(MIN_SIDE / FRAME)
    if len(L) < 2 * k:
        return 0.0
    return float(max(abs(np.median(L[:3]) - np.median(L[k:])), abs(np.median(L[-3:]) - np.median(L[:-k]))))

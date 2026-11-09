"""Teach Tickover a new question: named sound states for one machine, on the same frozen embeddings.

Drift answers "has it changed?". Some owners have a sharper question: "is the pump running dry?",
"is the compressor loaded or idling?". The agent turns that question into labels, asks for a few
recordings of each state, and builds a detector here. No training: each label is a memory bank
of CED + mn10 features, calibrated exactly like a drift baseline (leave one out, robust sigmas).

  classify   scores a recording against every label bank. The answer is the label it sits
             closest to in that label's own sigmas. If it is 3 sigmas or more from every label,
             the answer is "unknown": a state nobody taught.
  test       holds out one whole recording at a time (all its 10 s segments together, so a long
             recording cannot vouch for itself), rebuilds that label's calibration without it,
             and classifies it. It reports n of N recordings right, never a percentage alone.
  active     a detector switches on only when the held-out test could not be luck: 2+ labels,
             4+ separate recordings per label, 8+ held-out recordings in all, 0.8 or more of them
             right, a result that guessing would produce less than 1 time in 50, and a median
             gap of at least 1 sigma between the right state and the next. Below that it says
             so and stays off. Three recordings per state cannot pass: 6 of 6 happens by luck
             too often to trust.
  uncertain  a recording that sits within 1 sigma of two states gets no label.

A label is the owner's name for a state. It is not a diagnosis.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from . import audio, engine, store

MIN_RECORDINGS = 4
RECOMMENDED = 5
MIN_HELD_OUT = 8
MIN_ACCURACY = 0.8
MAX_LUCK = 0.02     # chance that guessing does this well
MIN_MARGIN = 1.0    # sigmas between the closest state and the next
UNKNOWN = engine.DRIFT


def _dir(name: str, create: bool = True):
    d = store.data_dir() / "detectors" / store.slug(name)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _meta(name: str) -> dict:
    f = _dir(name, False) / "detector.json"
    if not f.exists():
        raise KeyError(f"no detector named '{name}'. Teach it first with action='add'.")
    return json.loads(f.read_text())


def _save_meta(name: str, m: dict) -> None:
    (_dir(name) / "detector.json").write_text(json.dumps(m, indent=1))


def _load_label(name: str, label: str) -> tuple[list[dict], np.ndarray]:
    z = np.load(_dir(name, False) / f"{store.slug(label)}.npz")
    fr = np.split(z["frames"].astype(np.float32), np.cumsum(z["lens_f"])[:-1])
    return [{"pools": p, "frames": f} for p, f in zip(z["pools"], fr)], z["groups"]


def _save_label(name: str, label: str, F: list[dict], groups) -> None:
    np.savez_compressed(_dir(name) / f"{store.slug(label)}.npz",
                        pools=np.stack([f["pools"] for f in F]).astype(np.float32),
                        frames=np.concatenate([f["frames"] for f in F]).astype(np.float16),
                        lens_f=np.array([len(f["frames"]) for f in F]), groups=np.asarray(groups, np.int64))


def _features(paths: list[str]) -> tuple[list[dict], list[int], list[str]]:
    """Segment each recording; every segment carries the index of its recording (its group).
    Recordings and segments that fail the quality gate are left out, as in a drift baseline."""
    try:
        F, skipped = engine._usable(paths)
    except ValueError as exc:
        return [], [], [str(exc)]
    return [{"pools": f["pools"], "frames": f["frames"]} for f in F], [f["src"] for f in F], skipped


# ---------------------------------------------------------------- scoring
def _calibrate(F: list[dict], groups) -> tuple[np.ndarray, np.ndarray] | None:
    """Leave-one-recording-out raw scores of a label's own segments -> robust (med, mad)."""
    groups = np.asarray(groups)
    rows = []
    for i in range(len(F)):
        keep = [j for j in range(len(F)) if groups[j] != groups[i]]
        if keep:
            rows.append(engine._bank_scores(F, keep, F[i]))
    if len(rows) < 2:
        return None
    return engine._robust(np.stack(rows))


def _sigma(F: list[dict], keep: list[int], cal, q: dict) -> float:
    med, mad = cal
    return float(((engine._bank_scores(F, keep, q) - med) / mad)[:3].mean())


RESERVED = ("unknown", "uncertain", "not-ready", "not_ready")  # the answers a detector gives when it will not name a state
MIN_STATE = 0.75   # every state must be recognised at least this often in the held-out test


def _luck(right: int, total: int, labels: int) -> float:
    """Chance of getting `right` or more of `total` by guessing among `labels` states."""
    from math import comb
    p = 1.0 / labels
    return float(sum(comb(total, k) * p ** k * (1 - p) ** (total - k) for k in range(right, total + 1)))


def _decide(zs: dict[str, float]) -> dict:
    order = sorted(zs, key=zs.get)
    best = order[0]
    margin = zs[order[1]] - zs[best] if len(order) > 1 else None
    known = zs[best] < UNKNOWN
    label = "unknown" if not known else "uncertain" if margin is not None and margin < MIN_MARGIN else best
    return {"label": label, "closest": best,
            "sigma": {k: round(v, 2) for k, v in zs.items()},
            "margin_sigma": None if margin is None else round(margin, 2)}


# ---------------------------------------------------------------- public API
def add(name: str, label: str, paths: list[str], machine: str | None = None) -> dict:
    label = label.strip()
    if not label:
        raise ValueError("label is empty")
    if store.slug(label) in RESERVED:
        raise ValueError(f"'{label}' is a reserved answer. Name the state something else, e.g. 'odd noise'.")
    paths = list(dict.fromkeys(paths))
    try:
        m = _meta(name)
    except KeyError:
        m = {"name": name, "machine": machine, "labels": {}, "created_at": time.time(), "active": False}
    if machine:
        m["machine"] = machine
    Fn, gn, skipped = _features(paths)
    if not Fn:
        raise ValueError("no usable audio: " + "; ".join(skipped))
    # The same audio taught twice would sit on both sides of the held-out test. Keep each segment once.
    import hashlib
    seen, keep = set(m.get("hashes", [])), []
    for i, f in enumerate(Fn):
        h = hashlib.sha1(np.ascontiguousarray(f["pools"]).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    if len(keep) < len(Fn):
        skipped.append(f"{len(Fn) - len(keep)} segment(s) already taught to this detector, left out")
    Fn, gn = [Fn[i] for i in keep], [gn[i] for i in keep]
    if not Fn:
        raise ValueError("these recordings were already taught to this detector")
    m["hashes"] = sorted(seen)
    key = store.slug(label)
    spelled = {store.slug(k): k for k in m["labels"]}
    label = spelled.get(key, label)  # 'Running Dry' and 'running dry' are one state
    if key in spelled:
        F, g = _load_label(name, label)
        off = int(g.max()) + 1 if len(g) else 0
        F, g = F + Fn, list(g) + [x + off for x in gn]
    else:
        F, g = Fn, gn
    _save_label(name, label, F, g)
    m["labels"][label] = {"recordings": int(len(set(g))), "segments": len(F),
                          "sources": m["labels"].get(label, {}).get("sources", []) + [str(p) for p in paths]}
    m.pop("test", None)  # new data invalidates the last test
    m["active"] = False
    _save_meta(name, m)
    need = {k: max(0, MIN_RECORDINGS - v["recordings"]) for k, v in m["labels"].items()}
    ready = len(m["labels"]) >= 2 and not any(need.values())
    out = {"ok": True, "detector": name, "label": label, "labels": {k: v["recordings"] for k, v in m["labels"].items()},
           "skipped": skipped, "ready_to_test": ready}
    if ready:
        out["test"] = test(name)
    elif len(m["labels"]) < 2:
        out["message"] = (f"'{label}' has {m['labels'][label]['recordings']} recording(s). A detector needs at least two "
                          f"states. Ask for {RECOMMENDED} separate recordings of each state, from the same spot "
                          f"({MIN_RECORDINGS} is the minimum).")
    else:
        short = ", ".join(f"{k} needs {n} more" for k, n in need.items() if n)
        out["message"] = f"Not ready to test: {short}. Each recording should be about 10 s, from the same spot."
    return out


def test(name: str) -> dict:
    m = _meta(name)
    labels = list(m["labels"])
    if len(labels) < 2:
        raise ValueError("a detector needs at least two labels")
    banks = {lab: _load_label(name, lab) for lab in labels}
    full = {lab: _calibrate(*banks[lab]) for lab in labels}
    confusion = {a: {b: 0 for b in labels + ["uncertain", "unknown"]} for a in labels}
    per = {lab: [0, 0] for lab in labels}
    rows, right, total, margins = [], 0, 0, []
    for lab in labels:
        F, g = banks[lab]
        for grp in sorted(set(g.tolist())):
            held = [i for i in range(len(F)) if g[i] == grp]
            rest = [i for i in range(len(F)) if g[i] != grp]
            sub = [F[i] for i in rest]
            cal = _calibrate(sub, g[rest]) if len(set(g[rest].tolist())) >= 2 else None
            if cal is None:
                continue  # too few other recordings of this label to calibrate without this one
            zs = {}
            for other in labels:
                if other == lab:
                    zs[other] = float(np.mean([_sigma(sub, list(range(len(sub))), cal, F[i]) for i in held]))
                elif full[other] is not None:  # a state with one usable recording cannot be scored against
                    Fo, _ = banks[other]
                    zs[other] = float(np.mean([_sigma(Fo, list(range(len(Fo))), full[other], F[i]) for i in held]))
            if len(zs) < 2:
                continue
            d = _decide(zs)
            confusion[lab][d["label"]] += 1
            total += 1
            right += d["label"] == lab
            per[lab][1] += 1
            per[lab][0] += d["label"] == lab
            margins.append(min(v for k, v in zs.items() if k != lab) - zs[lab])
            rows.append({"true": lab, "recording": grp, "predicted": d["label"], "sigma": d["sigma"]})
    acc = right / total if total else 0.0
    enough = all(v["recordings"] >= MIN_RECORDINGS for v in m["labels"].values())
    # Guessing means always naming the state with the most recordings, so luck is measured against that.
    majority = max((t for _, t in per.values()), default=0)
    luck = _luck(right, total, total / majority) if total and majority else 1.0
    margin = float(np.median(margins)) if margins else 0.0
    by_guess = majority
    weak = [k for k, (r, t) in per.items() if t < MIN_RECORDINGS or r / t < MIN_STATE]
    active = bool(enough and not weak and total >= MIN_HELD_OUT and acc >= MIN_ACCURACY and luck < MAX_LUCK
                  and margin >= MIN_MARGIN)
    m["test"] = {"right": right, "total": total, "luck": round(luck, 4), "margin_sigma": round(margin, 2), "at": time.time()}
    m["active"] = active
    _save_meta(name, m)
    if active:
        msg = (f"Held-out test: {right} of {total} recordings right; guessing would get about {by_guess:.0f}. "
               f"The detector is on. Tell the owner the count, not a percentage, and that it rests on {total} recordings.")
    elif not enough or total < MIN_HELD_OUT:
        msg = (f"Held-out test so far: {right} of {total} right. Each state needs {MIN_RECORDINGS} separate recordings "
               f"({RECOMMENDED} is better) before the detector can switch on; fewer cannot rule out luck.")
    elif weak and acc >= MIN_ACCURACY and luck < MAX_LUCK:
        short = ", ".join(f"{k}: {per[k][0]} of {per[k][1]}" for k in weak)
        msg = (f"Held-out test: {right} of {total} right overall, but one state is not recognised well enough "
               f"({short}). The detector stays off. Add more separate recordings of that state.")
    elif acc < MIN_ACCURACY or luck >= MAX_LUCK:
        msg = (f"Held-out test: {right} of {total} right; guessing would get about {by_guess:.0f}. That is too close to "
               "luck, so the detector stays off. The states may not sound different enough from this spot. Try "
               "recordings closer to the part that changes, or more recordings of each state.")
    else:
        msg = (f"Held-out test: {right} of {total} right, but the states sit only {margin:.1f} sigma apart. The detector "
               "stays off: a small change in the recording would flip the answer. Record closer to the part that changes.")
    return {"ok": True, "detector": name, "right": right, "total": total, "active": active,
            "expected_by_guessing": round(by_guess, 1), "chance_of_luck": round(luck, 4), "median_margin_sigma": round(margin, 2),
            "per_state": {k: {"right": r, "total": t} for k, (r, t) in per.items()},
            "confusion": confusion, "held_out": rows, "message": msg}


def classify(name: str, path: str) -> dict:
    m = _meta(name)
    if len(m["labels"]) < 2:
        raise ValueError("a detector needs at least two states before it can answer")
    zs, segs = {}, audio.segments(audio.load(path, url_seconds=10.0))
    Q = [engine._features(s) for s in segs]
    for lab in m["labels"]:
        F, g = _load_label(name, lab)
        cal = _calibrate(F, g)
        if cal is None:
            raise ValueError(f"label '{lab}' needs at least two recordings")
        zs[lab] = float(np.mean([_sigma(F, list(range(len(F))), cal, q) for q in Q]))
    d = _decide(zs)
    out = {"ok": True, "detector": name, "machine": m.get("machine"), "active": m.get("active", False), **d,
           "tested": m.get("test"), "segments_scored": len(segs)}
    if not out["active"]:
        t = m.get("test")
        out["label"] = "not_ready"
        out["warning"] = ("This detector is off: " + (f"its held-out test got {t['right']} of {t['total']}" if t else
                                                         "it has not been tested yet") +
                          f". The closest state was '{d['closest']}', shown for debugging only. Do not report it.")
    out["how_to_read"] = ("sigma is distance from each taught state in that state's own spread. The label is the "
                          "closest state; 'unknown' means 3+ sigma from all of them; 'uncertain' means two states "
                          "are within 1 sigma, so no label is given. Labels are the owner's names "
                          "for states, not a diagnosis.")
    return out


def listing(machine: str | None = None) -> dict:
    root = store.data_dir() / "detectors"
    out = []
    for f in sorted(root.glob("*/detector.json")) if root.exists() else []:
        m = json.loads(f.read_text())
        if machine and store.slug(m.get("machine") or "") != store.slug(machine):
            continue
        out.append({"detector": m["name"], "machine": m.get("machine"), "active": m.get("active", False),
                    "labels": {k: v["recordings"] for k, v in m["labels"].items()}, "test": m.get("test")})
    return {"ok": True, "detectors": out}


def delete(name: str) -> dict:
    d = _dir(name, False)
    if not (d / "detector.json").exists():
        raise KeyError(f"no detector named '{name}'")
    shutil.rmtree(d)
    return {"ok": True, "deleted": name}


def active_for(machine: str) -> list[str]:
    try:
        return [d["detector"] for d in listing(machine)["detectors"] if d["active"]]
    except Exception:  # noqa: BLE001  a hint only; never break a check
        return []

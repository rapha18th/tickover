"""Measure tickover_verify on public recordings.

No-ops   two windows of the same machine in the same state. Any "changed" is a false alarm.
           same clip:   first 5 s against last 5 s of one recording (what a real no-op looks like)
           other clip:  5 s of one recording against 5 s of another recording of the same machine
Starts   room tone (synthetic pink noise at -50 dBFS) to a machine, and a second machine
         starting next to one already running (fan alone, then fan plus pump).
Stops    the starts reversed.
Faults   normal to abnormal of the same machine. Verify is not built for this; reported so
         nobody assumes it.
Clip     the same pairs spliced into one 10 s recording, and whole unspliced recordings as
         no-ops, through the change-point search.

Prints, per case, how many read as changed, how many confirm in the expected direction,
and the tone and level distributions.
Run: python bench/verify_bench.py [n_per_case]
"""
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tickover_core import audio, verify  # noqa: E402

D = os.environ.get("TICKOVER_DATA", r"C:\Users\USER\Projects\earshot-data")
SR = audio.SR
H = 5 * SR


def room(n, seed, db=-50.0):
    rng = np.random.default_rng(seed)
    w = np.fft.rfft(rng.standard_normal(n))
    w /= np.sqrt(np.arange(len(w)) + 1.0)
    x = np.fft.irfft(w, n)
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)).astype(np.float32)


def files(*parts, n=40):
    return sorted(glob.glob(os.path.join(D, *parts)))[:n]


SETS = {
    "pump": files("dcase20", "pump", "train", "*.wav"),
    "fan": files("dcase20", "fan", "train", "*.wav"),
    "pump_abn": files("dcase20", "pump", "test", "anomaly_*.wav"),
    "car": files("engines", "hf_car", "MyDataSet", "healthy_*.wav"),
}
_x = {}


def wav(p):
    if p not in _x:
        _x[p] = audio.load(p)
    return _x[p]


def pairs(n, rng):
    """(case, kind, before, after) with 5 s windows. kind: noop | start | stop | fault."""
    out = []
    for name in ("pump", "fan", "car"):
        fs = [f for f in SETS[name] if len(wav(f)) >= 2 * H]
        for i in range(min(n, len(fs))):
            a = wav(fs[i])
            b = wav(fs[(i + 7) % len(fs)])
            out.append((f"{name}: same clip, first half vs second half", "noop", a[:H], a[H:2 * H]))
            out.append((f"{name}: two recordings, same state", "noop", a[:H], b[:H]))
            r = room(H, i)
            out.append((f"room tone -> {name}", "start", r, a[:H]))
            out.append((f"{name} -> room tone", "stop", a[:H], r))
    for i in range(n):
        out.append(("room tone, nothing switched (the relay failed)", "noop", room(H, 500 + i), room(H, 900 + i)))
    fan = [f for f in SETS["fan"] if len(wav(f)) >= 2 * H]
    pump = [f for f in SETS["pump"] if len(wav(f)) >= 2 * H]
    for i in range(min(n, len(fan), len(pump))):
        f, p = wav(fan[i]), wav(pump[i])
        out.append(("fan running -> fan + pump (second machine starts)", "start", f[:H], f[H:2 * H] + p[:H]))
        out.append(("fan + pump -> fan (second machine stops)", "stop", f[:H] + p[:H], f[H:2 * H]))
    abn = [f for f in SETS["pump_abn"] if len(wav(f)) >= H]
    for i in range(min(n, len(abn), len(pump))):
        out.append(("pump normal -> pump abnormal (not what verify is for)", "fault", wav(pump[i])[:H], wav(abn[i])[:H]))
    return out


def summarise(rows, clip):
    by = {}
    for case, kind, m in rows:
        by.setdefault((case, kind), []).append(m)
    rep = []
    for (case, kind), v in by.items():
        want = {"start": "louder", "stop": "quieter"}.get(kind)
        dec = [verify.decide(m, clip) for m in v]
        tone = np.array([m["tone_sigma"] for m in v])
        ls = np.array([m["level_sigma"] for m in v])
        db = np.array([m["level_delta_db"] for m in v])
        row = {"case": case, "kind": kind, "n": len(v),
               "changed": f"{sum(c for c, _ in dec)}/{len(v)}",
               "tone_med": round(float(np.median(tone)), 1), "tone_min": round(float(tone.min()), 1),
               "tone_max": round(float(tone.max()), 1),
               "level_sigma_med": round(float(np.median(ls)), 1), "level_sigma_min": round(float(ls.min()), 1),
               "level_sigma_max": round(float(ls.max()), 1), "level_db_med": round(float(np.median(db)), 1)}
        if want:
            row["confirmed"] = f"{sum(1 for c, d in dec if c and d == want)}/{len(v)}"
        rep.append(row)
    return rep


def run(n=20):
    P = pairs(n, random.Random(0))
    two, one = [], []
    for case, kind, a, b in P:
        two.append((case, kind, verify.measure(a, b)))
        one.append((case, kind, verify.search(np.concatenate([a, b]))[1]))
    # whole unspliced recordings through the change-point search: the honest no-op for clip mode
    for name in ("pump", "fan", "car"):
        for p in [f for f in SETS[name] if len(wav(f)) >= 2 * H][:n]:
            one.append((f"{name}: one unspliced recording", "noop", verify.search(wav(p)[: 2 * H])[1]))
    for secs in (9, 12, 17):
        for i in range(n):
            one.append((f"room tone, one {secs} s recording, nothing switched", "noop", verify.search(room(secs * SR, 700 + i))[1]))
    for name in ("pump", "fan"):
        for p in [f for f in SETS[name] if len(wav(f)) >= 2 * H][:n]:
            one.append((f"{name}: 9.5 s of one recording", "noop", verify.search(wav(p)[: int(9.5 * SR)])[1]))
    return {"before_after": summarise(two, False), "clip": summarise(one, True)}


if __name__ == "__main__":
    r = run(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
    import os
    json.dump(r, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_results.json"), "w"), indent=1)
    for mode, rows in r.items():
        print(f"\n== {mode} ==")
        for row in rows:
            print(json.dumps(row))

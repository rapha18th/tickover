"""Self-test with synthetic machine sound, so anyone can verify Nzeve in a minute with no recordings.

A "fan" is a rotor hum (fundamental plus harmonics with slow wobble) in broadband noise.
Healthy clips vary in speed, phase and noise. The faulty clip adds what a failing bearing
adds: a periodic click train and a rising high-frequency whine.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

SR = 16_000


def fan(seconds=10.0, rpm=1800.0, seed=0, fault=False):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = rpm / 60 * 7 * (1 + 0.004 * rng.standard_normal())       # 7-blade pass frequency
    wob = 1 + 0.002 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6.28))
    x = sum((0.5 / h) * np.sin(2 * np.pi * f0 * h * wob * t + rng.uniform(0, 6.28)) for h in range(1, 9))
    x += 0.08 * rng.standard_normal(len(t))
    if fault:
        clicks = np.zeros_like(t)
        period = int(SR / (rpm / 60 * 3.2))                         # outer-race defect rate
        clicks[::period] = 1.0
        ring = np.exp(-np.arange(200) / 25) * np.sin(2 * np.pi * 3100 * np.arange(200) / SR)
        x += 0.9 * np.convolve(clicks, ring, mode="same")
        x += 0.12 * np.sin(2 * np.pi * (5200 + 300 * t / seconds) * t)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def run(verbose=True) -> dict:
    import soundfile as sf
    from . import engine
    d = tempfile.mkdtemp(prefix="nzeve_selftest_")
    healthy = []
    for i in range(6):
        p = os.path.join(d, f"healthy_{i}.wav")
        sf.write(p, fan(rpm=1800 + 15 * i, seed=i), SR)
        healthy.append(p)
    ok_p, bad_p = os.path.join(d, "check_ok.wav"), os.path.join(d, "check_fault.wav")
    sf.write(ok_p, fan(rpm=1806, seed=42), SR)
    sf.write(bad_p, fan(rpm=1806, seed=43, fault=True), SR)
    name = "nzeve selftest fan"
    e = engine.enroll(name, healthy)
    a = engine.check(name, ok_p)
    b = engine.check(name, bad_p)
    passed = a["verdict"] == "healthy" and b["verdict"] == "drift"
    out = {"ok": passed, "enroll": e["message"], "healthy_clip": {"verdict": a["verdict"], "drift_sigma": a["drift_sigma"]},
           "faulty_clip": {"verdict": b["verdict"], "drift_sigma": b["drift_sigma"], "most_changed_band": b["most_changed_band"]},
           "card": b["card"]}
    if verbose:
        print(("PASS" if passed else "FAIL"), out)
    return out

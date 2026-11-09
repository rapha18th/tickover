"""Self-test with synthetic machine sound, so anyone can verify Tickover in a minute with no recordings.

A "fan" is a rotor hum (fundamental plus harmonics with slow wobble) in broadband noise.
Healthy clips vary in speed, phase and noise. The faulty clip adds what a failing bearing
adds: a periodic click train and a rising high-frequency whine. The room is quiet noise.

Three checks: a healthy clip reads healthy, the faulty clip reads drift, and room then fan
is confirmed as a start. The test machine is removed afterwards.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

SR = 16_000
NAME = "tickover selftest fan"


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


def room(seconds=6.0, seed=0, dbfs=-50.0):
    """Quiet room tone: low-passed noise at about -50 dBFS."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(int(seconds * SR)))
    x -= np.convolve(x, np.ones(800) / 800, mode="same")           # take out the drift
    x += 0.3 * np.std(x) * rng.standard_normal(len(x))
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (dbfs / 20)).astype(np.float32)


def environment() -> dict:
    from . import audio, store
    try:
        import sounddevice as sd
        mic = bool(sd.query_devices(kind="input"))
    except Exception:  # noqa: BLE001
        mic = False
    return {"data_dir": str(store.data_dir()), "ffmpeg": bool(audio.ffmpeg()), "microphone": mic,
            "notes": ([] if audio.ffmpeg() else ["No ffmpeg: WAV, FLAC, OGG and Opus voice notes work; M4A, MP3, video "
                                                  "and RTSP streams need it. pip install imageio-ffmpeg is enough."])
                     + ([] if mic else ["No microphone here: tickover_listen and microphone verify are off; files and URLs work."])}


def run(verbose=True) -> dict:
    import soundfile as sf
    from . import engine, store, verify
    d = tempfile.mkdtemp(prefix="tickover_selftest_")
    try:
        healthy = []
        for i in range(6):
            p = os.path.join(d, f"healthy_{i}.wav")
            sf.write(p, fan(rpm=1800 + 15 * i, seed=i), SR)
            healthy.append(p)
        ok_p, bad_p = os.path.join(d, "check_ok.wav"), os.path.join(d, "check_fault.wav")
        sf.write(ok_p, fan(rpm=1806, seed=42), SR)
        sf.write(bad_p, fan(rpm=1806, seed=43, fault=True), SR)
        png = str(store.data_dir() / "selftest.png")
        e = engine.enroll(NAME, healthy)
        a = engine.check(NAME, ok_p, out_png=os.path.join(d, "ok.png"))
        b = engine.check(NAME, bad_p, out_png=png)
        m = verify.measure(room(6.0, seed=7), fan(6.0, rpm=1806, seed=44))
        changed, direction = verify.decide(m)
        still, _ = verify.decide(verify.measure(fan(6.0, rpm=1806, seed=45), fan(6.0, rpm=1806, seed=46)))
    finally:
        store.forget(NAME)
        shutil.rmtree(d, ignore_errors=True)
    checks = {"healthy clip reads healthy": a.get("verdict") == "healthy",
              "faulty clip reads drift": b.get("verdict") == "drift",
              "a start is heard": bool(changed and direction == "louder"),
              "no change reads as no change": not still}
    passed = all(checks.values())
    out = {"ok": passed, "checks": checks, "enroll": e["message"],
           "healthy_clip": {"verdict": a.get("verdict"), "drift_sigma": a.get("drift_sigma")},
           "faulty_clip": {"verdict": b.get("verdict"), "drift_sigma": b.get("drift_sigma"),
                           "most_changed_band": b.get("most_changed_band")},
           "start": {"tone_sigma": round(m["tone_sigma"], 1), "level_delta_db": round(m["level_delta_db"], 1),
                     "level_sigma": round(m["level_sigma"], 1), "heard": direction},
           "environment": environment(), "card": png if os.path.exists(png) else None,
           "message": ("Tickover works here. " if passed else "Self-test failed: " + ", ".join(k for k, v in checks.items() if not v) + ". ")
                      + "The sounds are synthetic; the card shows the faulty fan against its healthy baseline."}
    if verbose:
        print(("PASS" if passed else "FAIL"), out)
    return out

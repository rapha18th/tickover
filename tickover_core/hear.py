"""Hear: what is in this recording, and is it good enough to use?

The first thing Tickover does with any sound. CED's 527 AudioSet classes give the top tags
("Engine", "Idling", "Mechanical fan"), each with the seconds where it is strongest, and the
quality gate says whether the clip is clean. No baseline, no setup: it works on the first
voice note. Tags are what the clip sounds like to a general audio model. They are context
for the conversation, never a diagnosis.
"""
from __future__ import annotations

import time

import numpy as np

from . import audio, encoder, quality, store

MAX_SECONDS = 60.0
FINE = 2.0  # seconds: resolution of the "strongest" spans
MACHINE = {"Engine", "Idling", "Mechanical fan", "Air conditioning", "Pump (liquid)", "Mechanisms", "Tools", "Power tool",
           "Drill", "Vehicle", "Motor vehicle (road)", "Car", "Truck", "Motorcycle", "Light engine (high frequency)",
           "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine knocking", "Engine starting",
           "Accelerating, revving, vroom", "Lawn mower", "Chainsaw", "Sewing machine", "Printer", "Vacuum cleaner",
           "Blender", "Hair dryer", "Hum", "Whir", "Rumble", "Vibration", "Buzz", "Static", "White noise", "Pink noise",
           "Mains hum", "Gears", "Sawing", "Sanding", "Jackhammer", "Aircraft", "Aircraft engine", "Propeller, airscrew",
           "Helicopter", "Train", "Motorboat, speedboat", "Electric toothbrush", "Microwave oven", "Steam", "Hiss",
           "Clicking", "Tick-tock", "Tick", "Squeak", "Rattle"}
QUIET = {"Silence"}
# Tags that read as a diagnosis. A healthy diesel often scores "Engine knocking". Never shown.
HIDE = {"Engine knocking", "Rattle", "Squeak", "Explosion", "Burst, pop", "Crack", "Breaking", "Smash, crash",
        "Fire", "Smoke detector, smoke alarm", "Alarm", "Siren"}


def windows(x: np.ndarray) -> list[np.ndarray]:
    """10 s windows with no overlap; a tail shorter than 1 s is dropped."""
    n = encoder.CLIP
    return [x[s:s + n] for s in range(0, len(x), n) if len(x[s:s + n]) >= audio.SR] or [x]


def tag_probs(x: np.ndarray) -> np.ndarray:
    enc = encoder.get()
    return np.stack([enc.probs(w) for w in windows(x)])


def hear(path: str, top: int = 5, card: bool = True, out_png: str | None = None) -> dict:
    x = audio.load(path, max_seconds=MAX_SECONDS, url_seconds=10.0)
    if len(x) < audio.SR:
        raise ValueError(f"clip is {len(x) / audio.SR:.1f} s; Tickover needs at least 1 s to hear anything")
    enc, names = encoder.get(), encoder.labels()
    P = tag_probs(x)
    q = quality.assess(x, P)
    mean = P.mean(0)
    order = [int(i) for i in np.argsort(-mean) if names[int(i)] not in QUIET | HIDE][:top]
    # finer windows say when each tag is strongest; only worth it when the clip is long enough to have a "when"
    step = int(FINE * audio.SR)
    fine = [x[s:s + step] for s in range(0, len(x) - step + 1, step)] if len(x) >= 2 * step else []
    Pf = np.stack([enc.probs(w) for w in fine]) if fine else None
    tags = []
    for i in order:
        t = {"label": names[i], "score": round(float(mean[i]), 2)}
        if Pf is not None and Pf[:, i].max() > 0.05:
            on = set(np.flatnonzero(Pf[:, i] >= 0.5 * Pf[:, i].max()).tolist())
            lo = hi = int(np.argmax(Pf[:, i]))
            while lo - 1 in on:
                lo -= 1
            while hi + 1 in on:
                hi += 1
            t["strongest_s"] = [round(lo * FINE, 1), round(min((hi + 1) * FINE, len(x) / audio.SR), 1)]
        tags.append(t)
    machine_like = [t["label"] for t in tags if t["label"] in MACHINE and t["score"] >= 0.08]
    out = {"ok": True, "verdict": "heard" if q["ok"] else "retake", "seconds": q["seconds"], "tags": tags,
           "sounds_like_a_machine": bool(machine_like), "quality": q, "source": str(path)}
    if card:
        from . import card as _card, engine
        png = out_png or str(store.data_dir() / f"hear_{int(time.time() * 1000)}.png")
        _card.hear_card(png, tags=tags, quality=q, mel=engine._logmel(x[: int(30 * audio.SR)]), seconds=len(x) / audio.SR)
        out["card"] = png
    if not q["ok"]:
        out["message"] = "Not a usable recording yet. " + q["fix"]
    elif machine_like:
        out["message"] = ("Clean clip. It sounds like " + ", ".join(t["label"].lower() for t in tags[:2]) + ". "
                          "If this is a machine the owner cares about, offer to remember it: ask for a name, then two more "
                          "recordings from the same spot, and call tickover_enroll.")
    else:
        out["message"] = "Clean clip. It sounds like " + ", ".join(t["label"].lower() for t in tags[:3]) + "."
    out["how_to_read"] = ("Tags are what a general audio model hears, each with a 0-1 score. They are context, not a "
                          "diagnosis, and weak on unusual machines. Say 'sounds like', never 'is'.")
    return out

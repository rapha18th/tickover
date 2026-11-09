"""The quality gate: is this recording good enough to judge a machine by?

A drift score on a bad recording is worse than no score. A clipped clip, a clip that is mostly
silence, or one with a person talking over the machine all look like "the machine changed".
The gate runs in front of every verdict. A recording that fails gets the verdict "retake" with
one plain fix, and is never written to the machine's history, so trends stay clean.

Signal checks use the waveform. Speech and wind come from CED's AudioSet class probabilities,
which the drift scorer has already computed. Thresholds were set against the benchmark
recordings (DCASE pumps and fans, phone recordings of car engines), none of which may fail.
"""
from __future__ import annotations

import numpy as np

from . import audio

MIN_SECONDS = 3.0
CLIP_PCT = 0.1        # percent of samples at full scale
SILENT_DBFS = -55.0   # a 50 ms frame below this is silence
SILENT_PCT = 30.0
TOO_QUIET_DBFS = -60.0
SPEECH, WIND = 0.30, 0.30
MUSIC_NOTE = 0.50     # advisory only: steady hums can read as music
I_SPEECH, I_MUSIC, I_WIND = 0, 137, 285

FIX = {
    "too_short": "Record at least 5 seconds; 10 is best.",
    "clipping": "The recording is distorted. Move the phone further from the machine and record again.",
    "too_quiet": "The recording is almost silent. Move closer and check the microphone is not covered.",
    "silence": "Much of the recording is silence. Record while the machine is running, without pauses.",
    "speech": "Someone is talking over the machine. Record again without speaking.",
    "wind": "Wind is hitting the microphone. Shield the phone or move out of the wind and record again.",
}


def signal(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float32)
    secs = len(x) / audio.SR
    n = int(0.05 * audio.SR)
    fr = x[: len(x) // n * n].reshape(-1, n).astype(np.float64) if len(x) >= n else np.zeros((1, n))
    db = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    return {"seconds": round(secs, 1), "level_dbfs": round(audio.level_db(x), 1) if len(x) else -120.0,
            "clipping_pct": round(float((np.abs(x) >= 0.999).mean() * 100), 3) if len(x) else 0.0,
            "silence_pct": round(float((db < SILENT_DBFS).mean() * 100), 1)}


def assess(x: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """x: the whole recording. probs: (527,) or (segments, 527) class probabilities, if already computed."""
    q = signal(x)
    reasons = []
    if q["seconds"] < MIN_SECONDS:
        reasons.append("too_short")
    if q["clipping_pct"] > CLIP_PCT:
        reasons.append("clipping")
    if q["level_dbfs"] < TOO_QUIET_DBFS:
        reasons.append("too_quiet")
    elif q["silence_pct"] > SILENT_PCT:
        reasons.append("silence")
    if probs is not None:
        p = np.atleast_2d(np.asarray(probs, np.float32))
        q["speech"], q["wind"], q["music"] = (round(float(p[:, i].max()), 2) for i in (I_SPEECH, I_WIND, I_MUSIC))
        if q["speech"] > SPEECH:
            reasons.append("speech")
        if q["wind"] > WIND:
            reasons.append("wind")
        if q["music"] > MUSIC_NOTE:
            q["note"] = "This may contain music. If music was playing, record again without it."
    q["ok"] = not reasons
    q["reasons"] = reasons
    if reasons:
        q["fix"] = FIX[reasons[0]]
    return q


def segment_reason(probs: np.ndarray) -> str | None:
    """For enrolling long recordings: the reason one 10 s segment should be left out, or None."""
    if probs[I_SPEECH] > SPEECH:
        return "speech"
    if probs[I_WIND] > WIND:
        return "wind"
    return None

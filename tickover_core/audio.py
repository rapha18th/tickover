"""Audio intake: any file a chat app or phone produces becomes 16 kHz mono float32.

ffmpeg handles every container (Telegram .oga/.ogg Opus, WhatsApp .opus, iPhone .m4a,
video). Without ffmpeg, soundfile reads WAV, FLAC and OGG Vorbis.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

SR = 16_000


def load(path: str, max_seconds: float = 600.0) -> np.ndarray:
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-nostdin", "-v", "error", "-i", str(path), "-t", str(max_seconds),
               "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
        out = subprocess.run(cmd, capture_output=True, check=False, timeout=120)
        if out.returncode == 0 and out.stdout:
            return np.frombuffer(out.stdout, dtype=np.float32).copy()
    import soundfile as sf
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot decode {path}: install ffmpeg for Opus/M4A/video ({exc})") from exc
    x = x.mean(axis=1)
    if sr != SR:
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x[: int(max_seconds * SR)]


def segments(x: np.ndarray, seconds: float = 10.0, min_seconds: float = 2.0) -> list[np.ndarray]:
    """Cut a long recording into clip-length segments, so one minute of healthy sound enrolls six clips."""
    n, m = int(seconds * SR), int(min_seconds * SR)
    if len(x) < m:
        raise ValueError(f"clip is {len(x)/SR:.1f} s; Tickover needs at least {min_seconds:.0f} s of sound")
    if len(x) <= n * 1.5:
        return [x]
    return [x[s:s + n] for s in range(0, len(x) - m + 1, n) if len(x[s:s + n]) >= m]


def level_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9))


def record(seconds: float, device: str | int | None = None) -> np.ndarray:
    import sounddevice as sd
    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=device)
    sd.wait()
    return x[:, 0]

"""Audio intake: any file, phone recording or live stream becomes 16 kHz mono float32.

A source is a file path or a URL. ffmpeg handles every container (Telegram .oga/.ogg Opus,
WhatsApp .opus, iPhone .m4a, video) and every live source (an RTSP camera, an HTTP audio
stream, an ESP32 that serves a WAV). Without ffmpeg, soundfile reads WAV, FLAC and OGG, and
a plain HTTP download covers devices that serve one of those.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import time

import numpy as np

SR = 16_000
URL_SCHEMES = ("http://", "https://", "rtsp://", "rtsps://")
URL_SECONDS = 30.0          # how much of a live stream to take when the caller does not say
URL_MAX_BYTES = 40_000_000  # cap on a plain HTTP download


def is_url(path) -> bool:
    return str(path).strip().lower().startswith(URL_SCHEMES)


def ffmpeg() -> str | None:
    """ffmpeg on PATH, or the binary the imageio-ffmpeg wheel ships, or None."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001  not installed, or no binary for this platform
        return None


def _finite(x: np.ndarray, path) -> np.ndarray:
    if not np.isfinite(x).all():
        raise ValueError(f"{path} has corrupt samples (NaN or Inf). Record it again.")
    return x


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    x = x.mean(axis=1)
    if sr != SR:
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def _http(url: str, seconds: float) -> np.ndarray:
    """No ffmpeg: download what an http(s) device serves and let soundfile read it."""
    import urllib.request
    import soundfile as sf
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"cannot read {url}: install ffmpeg for RTSP and other live streams")
    buf, t0 = bytearray(), time.time()
    with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310 - scheme checked above
        while len(buf) < URL_MAX_BYTES and time.time() - t0 < seconds + 20:
            chunk = r.read(65536)
            if not chunk:
                break
            buf += chunk
    try:
        x, sr = sf.read(io.BytesIO(bytes(buf)), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot decode {url}: it did not return WAV, FLAC or OGG. Install ffmpeg for other formats ({exc})") from exc
    return _finite(_resample(x, sr)[: int(seconds * SR)], url)


def load(path: str, max_seconds: float = 600.0, url_seconds: float | None = None) -> np.ndarray:
    """Decode a file (up to max_seconds) or capture a URL (url_seconds, default 30)."""
    path = str(path).strip()
    url = is_url(path)
    secs = min(float(url_seconds or URL_SECONDS), max_seconds) if url else max_seconds
    ff = ffmpeg()
    if ff:
        cmd = [ff, "-nostdin", "-v", "error"]
        if url:
            cmd += ["-rtsp_transport", "tcp"] if path.lower().startswith("rtsp") else ["-rw_timeout", "15000000"]
        cmd += ["-i", path, "-t", str(secs), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
        try:
            out = subprocess.run(cmd, capture_output=True, check=False, timeout=secs + 60 if url else 120)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"timed out reading {path}") from exc
        if out.returncode == 0 and out.stdout:
            return _finite(np.frombuffer(out.stdout, dtype=np.float32).copy(), path)
        if url and not path.lower().startswith("http"):
            raise ValueError(f"cannot read {path}: {out.stderr.decode(errors='replace').strip()[-300:] or 'no audio'}")
    if url:
        return _http(path, secs)
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot decode {path}: install ffmpeg for M4A, MP3 or video ({exc})") from exc
    return _finite(_resample(x, sr)[: int(max_seconds * SR)], path)


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
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError("no microphone support here: pip install sounddevice, or pass a file or URL instead") from exc
    if isinstance(device, str) and device.strip().isdigit():
        device = int(device)
    try:
        x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=device)
        sd.wait()
    except Exception as exc:  # noqa: BLE001  PortAudio raises its own type
        raise RuntimeError(f"could not record from the microphone ({exc}). Pass a file or stream URL as source instead.") from exc
    return _finite(x[:, 0], "the microphone")


def capture(source: str | None, seconds: float, device: str | int | None = None) -> np.ndarray:
    """A window of sound from a file, a URL, or (source None) this computer's microphone."""
    if source:
        return load(source, max_seconds=600.0 if not is_url(source) else seconds, url_seconds=seconds)
    return record(seconds, device)

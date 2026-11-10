"""Audio intake: any file, phone recording or live stream becomes 16 kHz mono float32.

A source is a file path or a URL. ffmpeg handles every container (Telegram .oga/.ogg Opus,
WhatsApp .opus, iPhone .m4a, video) and every live source (an RTSP camera, an HTTP audio
stream, an ESP32 that serves a WAV). Without ffmpeg, soundfile reads WAV, FLAC and OGG, and
a plain HTTP download covers devices that serve one of those.

Every URL goes through sources.py first: the owner's allowed_sources list, a check of the resolved
address, and a connection pinned to that address. ffmpeg never dials an http(s) URL itself: Tickover
reads it and pipes the bytes in. RTSP goes through a relay that refuses redirects. Every ffmpeg run
gets -protocol_whitelist, a list of container demuxers it may use (no playlists, no concat), and a
minimal environment.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import threading
import time

import numpy as np

SR = 16_000
URL_SCHEMES = ("http://", "https://", "rtsp://", "rtsps://")
URL_SECONDS = 30.0          # how much of a live stream to take when the caller does not say
URL_MAX_BYTES = 40_000_000  # cap on a plain HTTP download
URL_TIMEOUT = 15.0          # seconds to wait for a device to answer


def _bridge(url: str) -> tuple[str, dict] | None:
    """('clip' or 'latest', query) when the URL points at a Tickover bridge endpoint."""
    from urllib.parse import parse_qs, urlsplit
    if not str(url).lower().startswith(("http://", "https://")):
        return None
    parts = urlsplit(str(url).strip())
    end = parts.path.rstrip("/").rsplit("/", 1)[-1].lower()
    return (end, parse_qs(parts.query)) if end in ("clip", "latest") else None


def redact_url(url) -> str:
    """The URL with any key= value hidden, for messages a person or another chat may see."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
    u = str(url)
    if not is_url(u):
        return u
    try:
        parts = urlsplit(u.strip())
        q = [(k, "***" if k.lower() == "key" else v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        netloc = parts.netloc
        if parts.password:   # rtsp://user:pass@camera
            netloc = netloc.replace(f":{parts.password}@", ":***@", 1)
        return urlunsplit(parts._replace(netloc=netloc, query=urlencode(q, safe="*")))
    except ValueError:
        return u.split("?", 1)[0]


def _wait(url: str) -> float:
    """A bridge /clip records before it answers, so wait for the recording too."""
    b = _bridge(url)
    if b and b[0] == "clip":
        try:
            secs = float((b[1].get("seconds") or ["10"])[0])
        except ValueError:
            secs = 10.0
        return min(max(secs, 3.0), 60.0) + 75.0 + URL_TIMEOUT  # recording, a busy microphone, the network
    return URL_TIMEOUT


def bridge_clip_url(url: str, seconds: float) -> str:
    """Ask a bridge /clip for the window the caller wants, unless the URL already says."""
    b = _bridge(url)
    if not b or b[0] != "clip" or "seconds" in b[1]:
        return url
    return f"{url}{'&' if '?' in url else '?'}seconds={int(math.ceil(seconds))}"


def bridge_latest(url: str) -> dict | None:
    """Ask a bridge /latest how old its newest clip is. None when the URL is no bridge or gives no age.

    The device timer is expected to write a clip every X-Tickover-Every seconds. A clip older than
    three periods (at least 15 minutes) means the device stopped recording.
    """
    from . import sources
    b = _bridge(url)
    if not b or b[0] != "latest":
        return None
    try:
        with sources.open_url(url, method="HEAD", timeout=URL_TIMEOUT) as r:   # allowlist, pinned address, checked redirects
            age = float(r.headers.get("X-Tickover-Age"))
            every = float(r.headers.get("X-Tickover-Every") or 0)
            at = r.headers.get("X-Tickover-Recorded-At")
            at = float(at) if at else None
    except Exception:  # noqa: BLE001  no age, a bad header, or no answer: the load that follows reports it
        return None
    if not math.isfinite(age) or age < 0:
        return None
    if not math.isfinite(every) or every < 0:
        every = 0.0
    if at is not None and not math.isfinite(at):
        at = None
    limit = max(3 * every, 900.0) if every > 0 else 3600.0
    return {"age_s": age, "every_s": every, "limit_s": limit, "stale": age > limit, "recorded_at": at}


BRIDGE_PHRASES = ("no clip yet", "no timed recordings", "unknown path", "the microphone could not record",
                  "the device gave silence")


def _device_error(url: str, exc: Exception) -> ValueError:
    """One plain sentence for a device that failed to give audio. The device's own words pass only when known."""
    import urllib.error
    url = redact_url(url)
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            import json
            detail = json.loads(exc.read(4096).decode(errors="replace") or "{}").get("error", "")
            detail = " ".join(str(detail).split())[:160]
        except Exception:  # noqa: BLE001
            detail = ""
        if not detail.lower().startswith(BRIDGE_PHRASES):
            detail = ""
        if exc.code == 401:
            return ValueError(f"The listening device at {url} refused the request. Add ?key=<its key> to the URL.")
        if exc.code == 503 and detail.lower().startswith("the device gave silence"):
            return ValueError(f"The listening device at {url} recorded silence. {detail}")
        if exc.code == 503:
            return ValueError(f"The listening device at {url} is busy with another recording. Try again in a minute.")
        return ValueError(f"The listening device at {url} answered {exc.code}. {detail}".strip())
    return ValueError(f"The listening device at {url} is not answering.")


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
    import soundfile as sf
    from . import sources
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"cannot read {redact_url(url)}: install ffmpeg for RTSP and other live streams")
    buf = bytearray()
    try:
        with sources.open_url(url, timeout=_wait(url)) as r:   # allowlist, pinned address, checked redirects
            t0 = time.time()   # a bridge /clip answers only after it has recorded
            while len(buf) < URL_MAX_BYTES and time.time() - t0 < seconds + 20:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
    except sources.SourceRefused:
        raise
    except Exception as exc:  # noqa: BLE001  refused, unreachable, timed out, or an HTTP error
        raise _device_error(url, exc) from exc
    if not buf:
        raise ValueError(f"The listening device at {redact_url(url)} is not answering.")
    try:
        x, sr = sf.read(io.BytesIO(bytes(buf)), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot decode {redact_url(url)}: it did not return WAV, FLAC or OGG. Install ffmpeg for other formats ({exc})") from exc
    return _finite(_resample(x, sr)[: int(seconds * SR)], redact_url(url))


# Demuxers ffmpeg may use on a file or a piped stream: audio and video containers only. No hls, dash,
# concat, image2, sdp or rtp, so a crafted file cannot make ffmpeg open other files or hosts.
SAFE_FORMATS = ("wav,w64,mp3,ogg,flac,aac,mov,matroska,webm,amr,amrnb,amrwb,aiff,caf,au,avi,asf,mpegts,mpeg,flv,"
                "ac3,eac3,dts,wv,ape,tta,loas")
WHITELIST_FILE = "file,pipe"
WHITELIST_PIPE = "pipe"
WHITELIST_RTSP = "rtsp,tcp"   # -rtsp_transport tcp, through the relay on 127.0.0.1 (sources.RtspGuard)


def ffmpeg_env() -> dict:
    """The environment ffmpeg runs with: enough to start on Linux, macOS and Windows, nothing else."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
    if os.name == "nt":   # Windows needs SYSTEMROOT to start a process; TEMP for scratch files
        keep += ("USERPROFILE", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
    return {k: os.environ[k] for k in keep if os.environ.get(k)}


def _decode_args(secs: float) -> list[str]:
    return ["-t", str(secs), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]


def _ffmpeg_file(ff: str, path: str, secs: float) -> subprocess.CompletedProcess:
    cmd = [ff, "-nostdin", "-v", "error", "-protocol_whitelist", WHITELIST_FILE, "-format_whitelist", SAFE_FORMATS,
           "-i", path, *_decode_args(secs)]
    return subprocess.run(cmd, capture_output=True, check=False, timeout=120, stdin=subprocess.DEVNULL, env=ffmpeg_env())


def _ffmpeg_pipe(ff: str, resp, secs: float, deadline: float) -> tuple[int, bytes, bytes]:
    """Feed an open http(s) response into ffmpeg on stdin. ffmpeg itself opens nothing but its pipes."""
    cmd = [ff, "-v", "error", "-protocol_whitelist", WHITELIST_PIPE, "-format_whitelist", SAFE_FORMATS,
           "-i", "pipe:0", *_decode_args(secs)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ffmpeg_env())
    out, err = [], []

    def feed():
        sent = 0
        try:
            while sent < URL_MAX_BYTES and time.time() < deadline:
                chunk = resp.read(65536)
                if not chunk:
                    break
                proc.stdin.write(chunk)
                sent += len(chunk)
        except (OSError, ValueError):
            pass   # ffmpeg has enough and closed its end, or the device stopped
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True),
               threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)]
    for t in threads:
        t.start()
    try:
        proc.wait(timeout=max(deadline - time.time(), 1.0) + 15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    finally:
        resp.close()
    for t in threads:
        t.join(timeout=5)
    return proc.returncode, b"".join(out), b"".join(err)


_UNREACHABLE = ("connection refused", "timed out", "no route", "unreachable", "failed to resolve",
                "name or service not known", "connection reset", "end of file")


def _load_url(ff: str | None, url: str, secs: float) -> np.ndarray:
    from . import sources
    try:
        target = sources.check(url)
    except LookupError as exc:
        raise ValueError(f"The listening device at {redact_url(url)} is not answering ({exc}).") from exc
    if not ff:
        return _http(url, secs)
    if target.scheme in ("http", "https"):
        try:
            resp = sources.open_url(url, timeout=_wait(url))
        except sources.SourceRefused:
            raise
        except Exception as exc:  # noqa: BLE001  refused, unreachable, timed out, or an HTTP error
            raise _device_error(url, exc) from exc
        code, out, err = _ffmpeg_pipe(ff, resp, secs, time.time() + secs + _wait(url) + 20)
        if code == 0 and out:
            return _finite(np.frombuffer(out, dtype=np.float32).copy(), redact_url(url))
        msg = err.decode(errors="replace").strip()[-300:]
        raise ValueError(f"cannot decode {redact_url(url)}: {msg or 'no audio'}")
    guard = sources.RtspGuard(target, timeout=URL_TIMEOUT)
    try:
        cmd = [ff, "-nostdin", "-v", "error", "-protocol_whitelist", WHITELIST_RTSP, "-format_whitelist", "rtsp",
               "-rtsp_transport", "tcp", "-i", guard.local_url(), *_decode_args(secs)]
        try:
            out = subprocess.run(cmd, capture_output=True, check=False, timeout=secs + _wait(url) + 45,
                                 stdin=subprocess.DEVNULL, env=ffmpeg_env())
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"The listening device at {redact_url(url)} is not answering.") from exc
    finally:
        guard.close()
    if guard.refused:
        raise sources.SourceRefused(f"{redact_url(url)}: {guard.refused}. Tickover does not follow camera redirects. "
                                    "Give the camera's final address.")
    if out.returncode == 0 and out.stdout:
        return _finite(np.frombuffer(out.stdout, dtype=np.float32).copy(), redact_url(url))
    err = out.stderr.decode(errors="replace")
    if any(s in err.lower() for s in _UNREACHABLE):
        raise ValueError(f"The listening device at {redact_url(url)} is not answering.")
    err = err.strip().replace(guard.local_url(), redact_url(url))[-300:]
    raise ValueError(f"cannot read {redact_url(url)}: {err or 'no audio'}")


def load(path: str, max_seconds: float = 600.0, url_seconds: float | None = None) -> np.ndarray:
    """Decode a file (up to max_seconds) or capture an allowed URL (url_seconds, default 30)."""
    path = str(path).strip()
    ff = ffmpeg()
    if is_url(path):
        return _load_url(ff, path, min(float(url_seconds or URL_SECONDS), max_seconds))
    from . import sources
    sources.local_path(path)   # a Windows network path would dial out before ffmpeg even starts
    ff_err = ""
    if ff:
        try:
            out = _ffmpeg_file(ff, path, max_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"timed out reading {path}") from exc
        if out.returncode == 0 and out.stdout:
            return _finite(np.frombuffer(out.stdout, dtype=np.float32).copy(), path)
        ff_err = out.stderr.decode(errors="replace").strip()[-200:]
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        if ff:
            raise ValueError(f"cannot decode {path}: it is not an audio or video file Tickover reads ({ff_err or exc})") from exc
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
        source = bridge_clip_url(source, seconds) if is_url(source) else source
        return load(source, max_seconds=600.0 if not is_url(source) else seconds, url_seconds=seconds)
    return record(seconds, device)

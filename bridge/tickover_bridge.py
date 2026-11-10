#!/usr/bin/env python3
"""Tickover bridge: turn a spare device with a microphone into an ear Hermes can listen to.

An old Android phone (Termux), a Raspberry Pi with a USB microphone, or any spare computer.
One file. Python standard library only. Download it and run it:

    python3 tickover_bridge.py --key SECRET              # serve http://<this device>:8765, ?key=SECRET required
    python3 tickover_bridge.py --key SECRET --every 600  # also keep a fresh clip at /latest
    python3 tickover_bridge.py --open                    # no key: anyone on this network can listen

Endpoints:
    GET /clip?seconds=N   record N seconds now (default 10, 3 to 60) and return a WAV
    GET /latest           the newest clip from the --every timer, with X-Tickover-Age
    GET /health           JSON status
    GET /                 a short page that lists these URLs

Allow it once on the Hermes computer with the command it prints (hermes config set
plugins.entries.tickover.settings.allowed_sources "[192.168.1.20:8765]").
Then tell Hermes: "Watch the borehole pump from http://192.168.1.20:8765/clip every morning at 7."

The bridge records 16 kHz mono. It serves only audio it records now, or the files given with
--replay. No request path ever touches the filesystem.
"""
from __future__ import annotations

import argparse
import array
import hmac
import html
import io
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

VERSION = "1.0.0"
SR = 16000
MIN_S, MAX_S, DEFAULT_S = 3, 60, 10
TIMER_MIN = 10            # tickover_check compares 10-second clips, so timed clips are never shorter
SILENT_DBFS = -85.0       # a clip below this level, or with no sample above zero, is silence
MAX_CLIP_WAITERS = 3      # /clip requests waiting for the microphone at once
MAX_CONNECTIONS = 32      # open connections at once
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")
BACKENDS = ("termux", "arecord", "sounddevice", "ffmpeg")


class CaptureError(RuntimeError):
    """The microphone could not record. The message says why."""


class SilentError(CaptureError):
    """The microphone recorded, but only silence."""


# --------------------------------------------------------------------------- audio helpers

def wav_bytes(pcm: bytes, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def peak(pcm: bytes) -> int:
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    return max((abs(v) for v in a), default=0)


def level_dbfs(pcm: bytes) -> float | None:
    """RMS level of 16-bit PCM in dB relative to full scale."""
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":
        a.byteswap()
    if not a:
        return None
    step = max(1, len(a) // 200_000)  # a sample of the clip is enough for a level
    s = a[::step]
    rms = math.sqrt(sum(v * v for v in s) / len(s)) / 32768.0
    return round(20 * math.log10(rms + 1e-9), 1)


def _to_mono16(raw: bytes, width: int, channels: int, rate: int) -> array.array:
    """Any PCM WAV payload becomes 16 kHz mono int16 samples."""
    if width == 1:
        vals = [(b - 128) << 8 for b in raw]
    elif width == 2:
        a = array.array("h")
        a.frombytes(raw[: len(raw) - len(raw) % 2])
        if sys.byteorder == "big":
            a.byteswap()
        vals = a
    elif width == 3:
        vals = [int.from_bytes(raw[i:i + 3], "little", signed=True) >> 8 for i in range(0, len(raw) - 2, 3)]
    elif width == 4:
        a = array.array("i")
        a.frombytes(raw[: len(raw) - len(raw) % 4])
        if sys.byteorder == "big":
            a.byteswap()
        vals = [v >> 16 for v in a]
    else:
        raise CaptureError(f"unsupported sample width {width}")
    if channels > 1:
        vals = [sum(vals[i:i + channels]) // channels for i in range(0, len(vals) - channels + 1, channels)]
    if rate != SR and vals:
        n = int(len(vals) * SR / rate)
        ratio = rate / SR
        vals = [vals[min(int(i * ratio), len(vals) - 1)] for i in range(n)]
    return array.array("h", vals)


def read_wav(path: str) -> array.array:
    with wave.open(path, "rb") as w:
        return _to_mono16(w.readframes(w.getnframes()), w.getsampwidth(), w.getnchannels(), w.getframerate())


def find_ffmpeg() -> str | None:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:  # present when Tickover itself is installed on this computer
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _run(argv: list[str], timeout: float) -> bytes:
    """Run a fixed argv list. No shell, ever."""
    try:
        out = subprocess.run(argv, capture_output=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(f"{os.path.basename(argv[0])} did not finish in {timeout:.0f} s") from exc
    except OSError as exc:
        raise CaptureError(f"could not start {argv[0]}: {exc}") from exc
    if out.returncode != 0 or not out.stdout:
        err = out.stderr.decode(errors="replace").strip()[-300:] or f"exit code {out.returncode}"
        raise CaptureError(f"{os.path.basename(argv[0])} failed: {err}")
    return out.stdout


# --------------------------------------------------------------------------- capture backends
# Each backend returns (payload bytes, content type, raw 16 kHz mono PCM or None).

class Backend:
    name = "none"

    def record(self, seconds: int) -> tuple[bytes, str, bytes | None]:
        raise NotImplementedError


class Arecord(Backend):
    """Linux and Raspberry Pi through ALSA."""
    name = "arecord"

    def __init__(self, device: str | None):
        self.exe, self.device = shutil.which("arecord"), device

    def record(self, seconds):
        argv = [self.exe, "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-d", str(int(seconds))]
        if self.device:
            dev = self.device
            if dev.startswith("hw:"):   # hw: does not resample, and most USB microphones refuse 16 kHz
                dev = "plug" + dev
            argv += ["-D", dev]
        argv.append("-")
        pcm = _run(argv, seconds + 20)
        return wav_bytes(pcm), "audio/wav", pcm


class SoundDevice(Backend):
    """Any computer with PortAudio and the sounddevice package."""
    name = "sounddevice"

    def __init__(self, device: str | None):
        import sounddevice  # type: ignore  # noqa: F401
        self.device = int(device) if device and device.isdigit() else device

    def record(self, seconds):
        import sounddevice as sd  # type: ignore
        try:
            x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16", device=self.device)
            sd.wait()
        except Exception as exc:  # noqa: BLE001  PortAudio raises its own type
            raise CaptureError(f"sounddevice could not record: {exc}") from exc
        pcm = x.tobytes()
        return wav_bytes(pcm), "audio/wav", pcm


class Ffmpeg(Backend):
    """ffmpeg reads the system microphone: alsa or pulse on Linux, avfoundation on macOS, dshow on Windows."""
    name = "ffmpeg"

    def __init__(self, device: str | None, exe: str | None = None):
        self.exe = exe or find_ffmpeg()
        if sys.platform == "darwin":
            self.fmt, self.device = "avfoundation", device or ":0"
        elif sys.platform.startswith("win"):
            self.fmt, self.device = "dshow", device or self._first_dshow()
            if self.device and not self.device.startswith("audio="):
                self.device = "audio=" + self.device
        else:
            self.fmt = "pulse" if shutil.which("pactl") else "alsa"
            self.device = device or "default"

    def _first_dshow(self) -> str | None:
        """The first DirectShow audio input. Best effort."""
        try:
            out = subprocess.run([self.exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                                 capture_output=True, timeout=15, check=False, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in out.stderr.decode(errors="replace").splitlines():
            if "(audio)" in line and '"' in line:
                return line.split('"')[1]
        return None

    def record(self, seconds):
        if not self.device:
            raise CaptureError("ffmpeg found no microphone. Pass --device with the input name")
        argv = [self.exe, "-nostdin", "-hide_banner", "-v", "error", "-f", self.fmt, "-i", self.device,
                "-t", str(int(seconds)), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
        pcm = _run(argv, seconds + 30)
        return wav_bytes(pcm), "audio/wav", pcm


class Termux(Backend):
    """Android through Termux:API. The phone records AAC; ffmpeg turns it into WAV when installed."""
    name = "termux"

    def __init__(self, device: str | None):
        self.exe = shutil.which("termux-microphone-record")
        self.ff = shutil.which("ffmpeg")

    def record(self, seconds):
        tmp = tempfile.mkdtemp(prefix="tickover-")
        path = os.path.join(tmp, "clip.m4a")
        try:
            subprocess.run([self.exe, "-q"], capture_output=True, timeout=10, check=False, stdin=subprocess.DEVNULL)
            _run_start = subprocess.run(
                [self.exe, "-f", path, "-l", str(int(seconds)), "-e", "aac", "-r", str(SR), "-c", "1"],
                capture_output=True, timeout=20, check=False, stdin=subprocess.DEVNULL)
            if _run_start.returncode != 0:
                raise CaptureError("termux-microphone-record failed: "
                                   + (_run_start.stderr.decode(errors="replace").strip() or "is Termux:API installed?"))
            time.sleep(seconds + 1.0)
            last, deadline = -1, time.time() + 10
            while time.time() < deadline:  # wait until the file stops growing
                size = os.path.getsize(path) if os.path.exists(path) else 0
                if size and size == last:
                    break
                last = size
                time.sleep(0.7)
            subprocess.run([self.exe, "-q"], capture_output=True, timeout=10, check=False, stdin=subprocess.DEVNULL)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise CaptureError("the phone wrote no audio. Grant Termux:API the microphone permission")
            if self.ff:
                pcm = _run([self.ff, "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                            "-f", "s16le", "-"], 60)
                return wav_bytes(pcm), "audio/wav", pcm
            with open(path, "rb") as f:
                return f.read(), "audio/mp4", None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class Replay(Backend):
    """Serve existing recordings in place of a microphone. Cycles through a folder."""
    name = "replay"
    EXT = (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac")

    def __init__(self, source: str, realtime: bool = False):
        source = os.path.abspath(os.path.expanduser(source))
        if os.path.isdir(source):
            files = sorted(os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(self.EXT))
        elif os.path.isfile(source):
            files = [source]
        else:
            raise CaptureError(f"--replay: {source} is not a file or folder")
        self.ff = find_ffmpeg()
        self.clips: list[array.array] = []
        for f in files:
            try:
                self.clips.append(self._decode(f))
            except Exception as exc:  # noqa: BLE001  one bad file should not stop the others
                print(f"replay: skipping {os.path.basename(f)}: {exc}", file=sys.stderr)
        self.clips = [c for c in self.clips if len(c)]
        if not self.clips:
            raise CaptureError(f"--replay: no readable audio in {source}")
        self.realtime, self.i, self.pos = realtime, 0, 0
        self.names = [os.path.basename(f) for f in files]

    def _decode(self, path: str) -> array.array:
        if path.lower().endswith(".wav"):
            try:
                return read_wav(path)
            except (wave.Error, EOFError, CaptureError):
                pass  # float or compressed WAV: let ffmpeg try
        if not self.ff:
            raise CaptureError("not a plain PCM WAV and no ffmpeg here")
        pcm = _run([self.ff, "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], 120)
        a = array.array("h")
        a.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        if sys.byteorder == "big":
            a.byteswap()
        return a

    def record(self, seconds):
        need, out = int(seconds * SR), array.array("h")
        while len(out) < need:
            clip = self.clips[self.i]
            take = clip[self.pos:self.pos + need - len(out)]
            out.extend(take)
            self.pos += len(take)
            if self.pos >= len(clip):
                self.i, self.pos = (self.i + 1) % len(self.clips), 0
        if self.realtime:
            time.sleep(seconds)
        if sys.byteorder == "big":
            out.byteswap()
        pcm = out.tobytes()
        return wav_bytes(pcm), "audio/wav", pcm


def detect_backend(choice: str | None, device: str | None) -> Backend:
    """termux, then arecord, then sounddevice, then ffmpeg. --backend forces one."""
    order = [choice] if choice else list(BACKENDS)
    errors = []
    for name in order:
        if name == "termux":
            if shutil.which("termux-microphone-record"):
                return Termux(device)
            errors.append("termux: termux-microphone-record not found (pkg install termux-api)")
        elif name == "arecord":
            if shutil.which("arecord"):
                return Arecord(device)
            errors.append("arecord: not found (sudo apt install alsa-utils)")
        elif name == "sounddevice":
            try:
                return SoundDevice(device)
            except Exception as exc:  # noqa: BLE001  ImportError, or OSError when PortAudio is missing
                errors.append(f"sounddevice: {exc}")
        elif name == "ffmpeg":
            exe = find_ffmpeg()
            if exe:
                return Ffmpeg(device, exe)
            errors.append("ffmpeg: not found")
    raise CaptureError("no way to record on this device. " + "; ".join(errors)
                       + ". Use --replay FILE_OR_DIR to serve recordings instead.")


# --------------------------------------------------------------------------- state

class Bridge:
    def __init__(self, backend: Backend, name: str, key: str | None, every: float | None,
                 seconds: int, wait: float):
        self.backend, self.name, self.key = backend, name, key
        self.every, self.seconds, self.wait = every, seconds, wait
        self.lock = threading.Lock()          # one capture at a time
        self.waiters = threading.BoundedSemaphore(MAX_CLIP_WAITERS)
        self.latest: dict | None = None       # newest timer clip
        self.last_level: float | None = None
        self.last_clip_at: float | None = None
        self.last_error: str | None = None
        self.started = time.time()
        self.stop = threading.Event()

    def capture(self, seconds: int, block: bool = False) -> dict | None:
        """Record now. None means the microphone is busy with another capture."""
        if not self.lock.acquire(timeout=-1 if block else self.wait):
            return None
        try:
            payload, ctype, pcm = self.backend.record(seconds)
            level = level_dbfs(pcm) if pcm else None
            if pcm is not None and (not pcm or peak(pcm) == 0 or (level is not None and level < SILENT_DBFS)):
                raise SilentError("The device gave silence. On Android, keep Termux open on screen or check the "
                                  "microphone permission. Elsewhere, check the microphone is plugged in and not muted.")
            now = time.time()
            self.last_level, self.last_clip_at, self.last_error = level, now, None
            return {"data": payload, "type": ctype, "at": now, "level": level, "seconds": seconds}
        except CaptureError as exc:
            self.last_error = str(exc)
            raise
        finally:
            self.lock.release()

    def timer(self) -> None:
        while not self.stop.is_set():
            t0 = time.time()
            try:
                self.latest = self.capture(self.seconds, block=True)
                print(time.strftime("%H:%M:%S"), f"recorded {self.seconds} s for /latest", flush=True)
            except CaptureError as exc:
                print(time.strftime("%H:%M:%S"), "recording failed:", exc, file=sys.stderr, flush=True)
            self.stop.wait(max(1.0, self.every - (time.time() - t0)))

    def health(self) -> dict:
        now = time.time()
        return {"tickover_bridge": VERSION, "name": self.name, "backend": self.backend.name,
                "level_dbfs": self.last_level,
                "last_clip_age_s": None if self.last_clip_at is None else round(now - self.last_clip_at, 1),
                "uptime_s": round(now - self.started, 1),
                "every_s": self.every, "busy": self.lock.locked(), "last_error": self.last_error}


# --------------------------------------------------------------------------- HTTP

def make_handler(bridge: Bridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"TickoverBridge/{VERSION}"
        protocol_version = "HTTP/1.1"
        timeout = 30  # a stalled client cannot hold a thread forever

        def log_message(self, fmt, *args):  # keep the key out of logs: never the query, never an error's text
            path = urlsplit(getattr(self, "path", "") or "").path
            status = args[0] if fmt.startswith("code") else (args[1] if len(args) > 1 else "")
            cmd = getattr(self, "command", None) or "-"
            sys.stderr.write(f"{self.address_string()} {cmd} {path} {status}\n")

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None, head: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _json(self, code: int, obj: dict, extra: dict | None = None, head: bool = False):
            self._send(code, json.dumps(obj, indent=1).encode(), "application/json", extra, head)

        def _authorised(self, q: dict) -> bool:
            if not bridge.key:
                return True
            given = (q.get("key") or [""])[0] or self.headers.get("X-Tickover-Key", "")
            return hmac.compare_digest(given.encode(), bridge.key.encode())

        def do_HEAD(self):
            self._route(head=True)

        def do_GET(self):
            self._route(head=False)

        def do_POST(self):
            try:   # read a small body so the client sees the answer; then close, so nothing left is parsed
                n = int(self.headers.get("Content-Length") or 0)
                if 0 < n <= 65536:
                    self.rfile.read(n)
            except (ValueError, OSError):
                pass
            self.close_connection = True
            self._json(405, {"error": "only GET and HEAD are served"}, {"Allow": "GET, HEAD", "Connection": "close"})

        do_PUT = do_DELETE = do_PATCH = do_POST

        def _route(self, head: bool):
            parts = urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            try:
                q = parse_qs(parts.query, max_num_fields=8)
            except ValueError:
                return self._json(400, {"error": "too many query fields"}, head=head)
            if not self._authorised(q):
                return self._json(401, {"error": "this bridge needs ?key=..."}, head=head)
            if path == "/":
                return self._send(200, index_page(bridge, self.headers.get("Host")).encode(),
                                  "text/html; charset=utf-8", head=head)
            if path == "/health":
                return self._json(200, bridge.health(), head=head)
            if path == "/latest":
                return self._latest(head)
            if path == "/clip":
                if head:
                    return self._json(405, {"error": "HEAD /clip would not record. Use GET"}, {"Allow": "GET"}, head=True)
                return self._clip(q)
            return self._json(404, {"error": "unknown path. Try /clip, /latest or /health"}, head=head)

        def _latest(self, head: bool):
            clip = bridge.latest
            if not clip:
                msg = ("no clip yet. The first timed recording is on its way" if bridge.every else
                       "no timed recordings. Start the bridge with --every SECONDS, or use /clip")
                return self._json(404, {"error": msg}, head=head)
            age = time.time() - clip["at"]
            self._send(200, clip["data"], clip["type"], {
                "Last-Modified": formatdate(clip["at"], usegmt=True),
                "X-Tickover-Age": f"{age:.1f}",
                "X-Tickover-Recorded-At": f"{clip['at']:.3f}",
                "X-Tickover-Every": f"{bridge.every or 0:g}",
                "X-Tickover-Level": "" if clip["level"] is None else clip["level"]}, head=head)

        def _clip(self, q: dict):
            raw = (q.get("seconds") or [str(DEFAULT_S)])[0]
            try:
                val = float(raw)
            except ValueError:
                val = math.nan
            if not math.isfinite(val):
                return self._json(400, {"error": "seconds must be a number"})
            secs = max(MIN_S, min(MAX_S, int(round(val))))
            if not bridge.waiters.acquire(blocking=False):
                return self._json(503, {"error": "the microphone is busy with another recording"},
                                  {"Retry-After": max(MIN_S, bridge.seconds)})
            try:
                clip = bridge.capture(secs)
            except SilentError as exc:
                return self._json(503, {"error": str(exc)})
            except CaptureError as exc:
                return self._json(500, {"error": f"the microphone could not record: {exc}"})
            finally:
                bridge.waiters.release()
            if clip is None:
                return self._json(503, {"error": "the microphone is busy with another recording"},
                                  {"Retry-After": max(MIN_S, bridge.seconds)})
            self._send(200, clip["data"], clip["type"], {
                "Last-Modified": formatdate(clip["at"], usegmt=True),
                "X-Tickover-Age": "0.0",
                "X-Tickover-Recorded-At": f"{clip['at']:.3f}",
                "X-Tickover-Level": "" if clip["level"] is None else clip["level"]})

    return Handler


def index_page(bridge: Bridge, host: str | None) -> str:
    base = f"http://{html.escape(host or 'this-device:8765')}"
    k = "?key=..." if bridge.key else ""
    k2 = "&key=..." if bridge.key else ""
    h, _, p = (host or "").rpartition(":")
    allow = allow_command(h.strip("[]"), int(p)) if h and p.isdigit() else allow_command("THIS-DEVICE-IP", 8765)
    every = (f"Records {bridge.seconds} s every {bridge.every:g} s for /latest." if bridge.every
             else "No timer. Start with --every SECONDS to fill /latest.")
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Tickover bridge</title>
<body style="font-family:system-ui,sans-serif;max-width:40em;margin:2em auto;padding:0 16px;line-height:1.5">
<h1>Tickover bridge</h1>
<p>This device is an ear for <b>{html.escape(bridge.name)}</b>. Hermes listens through it.
Backend: {html.escape(bridge.backend.name)}. {html.escape(every)}</p>
<ul>
<li><code>{base}/clip?seconds=10{k2}</code> records now and returns a WAV</li>
<li><code>{base}/latest{k}</code> the newest timed clip</li>
<li><code>{base}/health{k}</code> status as JSON</li>
</ul>
<p>Allow this device once, on the computer Hermes runs on:
<code>{html.escape(allow)}</code><br>
This sets the whole list. <code>hermes config get plugins.entries.tickover.settings.allowed_sources</code> shows it.
To keep other devices, put them all in the brackets, comma separated.</p>
<p>Then tell Hermes: <i>Watch the {html.escape(bridge.name)} from {base}/clip{k} every morning at 7.</i></p>
</body>"""


# --------------------------------------------------------------------------- startup

def lan_addresses() -> list[str]:
    """The IPv4 addresses other devices can probably reach. Best effort."""
    ips: list[str] = []
    try:  # the address of the default route. A UDP connect sends no packet.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ips.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    ips = [ip for ip in ips if not ip.startswith("127.")]
    return ips or ["127.0.0.1"]


def advertise(port: int, name: str):
    """mDNS, only when the zeroconf package happens to be installed."""
    try:
        from zeroconf import ServiceInfo, Zeroconf  # type: ignore
        ip = lan_addresses()[0]
        safe = "".join(c for c in name if c.isalnum() or c in " -")[:40] or "tickover"
        info = ServiceInfo("_http._tcp.local.", f"Tickover {safe}._http._tcp.local.",
                           addresses=[socket.inet_aton(ip)], port=port,
                           properties={"path": "/clip", "tickover_bridge": VERSION})
        zc = Zeroconf()
        zc.register_service(info)
        return zc
    except Exception:  # noqa: BLE001  never required
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="tickover_bridge.py",
                                 description="Turn this device's microphone into an ear Hermes can listen to.")
    ap.add_argument("--host", default="0.0.0.0", help="address to serve on (default 0.0.0.0, all interfaces)")
    ap.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    ap.add_argument("--name", default=os.environ.get("TICKOVER_BRIDGE_NAME") or "machine", help="what this ear listens to, for example 'borehole pump'")
    ap.add_argument("--backend", choices=BACKENDS, help="force a capture backend (default: detect)")
    ap.add_argument("--device", help="microphone: ALSA device (plughw:1,0), dshow name, avfoundation ':0', or sounddevice index")
    ap.add_argument("--replay", metavar="FILE_OR_DIR", help="serve these recordings instead of a microphone")
    ap.add_argument("--realtime", action="store_true", help="with --replay, take as long as a real recording")
    ap.add_argument("--every", type=float, metavar="SECONDS", help="record on a timer and serve the newest at /latest")
    ap.add_argument("--seconds", type=int, default=DEFAULT_S,
                    help=f"length of each timed recording, {TIMER_MIN} to {MAX_S} (default {DEFAULT_S}). Checks need 10 s")
    ap.add_argument("--key", default=os.environ.get("TICKOVER_BRIDGE_KEY") or None,
                    help="require ?key=SECRET on every request (or set TICKOVER_BRIDGE_KEY)")
    ap.add_argument("--wait", type=float, default=75.0,
                    help="seconds a /clip request waits for a busy microphone before 503 (default 75)")
    ap.add_argument("--open", action="store_true", help="serve without a key: anyone on this network can listen")
    ap.add_argument("--quiet", action="store_true", help="do not print the start banner")
    a = ap.parse_args(argv)
    a.seconds = max(TIMER_MIN, min(MAX_S, a.seconds))
    if not a.key and not a.open and a.host not in LOCAL_HOSTS:
        ap.error("set --key SECRET, or pass --open to let anyone on this network listen")
    if a.every is not None and a.every < a.seconds + 1:
        ap.error("--every must be longer than --seconds")
    return a


class Server(ThreadingHTTPServer):
    """A thread per connection, up to MAX_CONNECTIONS. Past that, a new connection is closed at once."""

    def __init__(self, *args, **kw):
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(*args, **kw)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def build(a) -> tuple[Bridge, ThreadingHTTPServer]:
    backend = Replay(a.replay, a.realtime) if a.replay else detect_backend(a.backend, a.device)
    bridge = Bridge(backend, a.name, a.key, a.every, a.seconds, a.wait)
    server = Server((a.host, a.port), make_handler(bridge))
    server.daemon_threads = True
    server.request_queue_size = 16
    return bridge, server


def allow_command(host: str, port: int) -> str:
    """The command that adds this device to Tickover's allowed_sources setting on the Hermes computer."""
    entry = f"'[{host}]:{port}'" if ":" in host else f"{host}:{port}"
    return f'hermes config set plugins.entries.tickover.settings.allowed_sources "[{entry}]"'


def banner(a, bridge: Bridge, port: int) -> str:
    k = f"?key={a.key}" if a.key else ""
    hosts = lan_addresses() if a.host in ("0.0.0.0", "") else [a.host]
    lines = [f"Tickover bridge {VERSION}. Listening to '{a.name}' with {bridge.backend.name}.",
             "Reachable at:"]
    for h in hosts:
        lines.append(f"  http://{h}:{port}/clip{k}")
    try:
        host = socket.gethostname()
        lines.append(f"  http://{host}.local:{port}/clip{k}  (if your network resolves .local names; allow "
                     f"{host.lower()}.local:{port} as well to use it)")
    except OSError:
        pass
    if a.every:
        lines.append(f"Recording {a.seconds} s every {a.every:g} s. Newest at http://{hosts[0]}:{port}/latest{k}")
    lines += ["", "Once, on the computer Hermes runs on, allow this device:",
              f"  {allow_command(hosts[0], port)}",
              "  This sets the whole list. To see what is allowed now:",
              "  hermes config get plugins.entries.tickover.settings.allowed_sources",
              "  To keep those devices, put them all in the brackets, comma separated.",
              "", "Then tell Hermes:",
              f"  Watch the {a.name} from http://{hosts[0]}:{port}/clip{k} every morning at 7.", ""]
    if not a.key:
        lines.append("No --key set. Anyone on this network can listen. Add --key on shared networks.")
    return "\n".join(lines)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        bridge, server = build(a)
    except CaptureError as exc:
        print(f"tickover bridge: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"tickover bridge: cannot serve on {a.host}:{a.port}: {exc}", file=sys.stderr)
        return 2
    port = server.server_address[1]
    if not a.quiet:
        print(banner(a, bridge, port), flush=True)
    if a.every:
        threading.Thread(target=bridge.timer, daemon=True, name="timer").start()
    zc = advertise(port, a.name)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop.set()
        server.server_close()
        if zc:
            try:
                zc.close()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

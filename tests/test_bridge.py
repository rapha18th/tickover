"""The listening-device bridge in replay mode. No microphone, no network beyond localhost."""
import io
import json
import math
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE = os.path.join(ROOT, "bridge", "tickover_bridge.py")
SR = 16000


def _tone(path, seconds, freq, sr=SR, channels=1):
    n = int(seconds * sr)
    frames = b"".join(struct.pack("<" + "h" * channels, *([int(8000 * math.sin(2 * math.pi * freq * i / sr))] * channels))
                      for i in range(n))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(tmp_path, *extra):
    folder = tmp_path / "replay"
    folder.mkdir(exist_ok=True)
    _tone(folder / "a.wav", 4, 440)
    _tone(folder / "b.wav", 4, 880, sr=44100, channels=2)  # resampled and mixed down by the bridge
    port = _free_port()
    proc = subprocess.Popen([sys.executable, BRIDGE, "--replay", str(folder), "--host", "127.0.0.1",
                             "--port", str(port), "--name", "test pump", "--quiet", *extra],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc, base
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("bridge exited")
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("bridge did not start")


@pytest.fixture
def bridge(tmp_path):
    proc, base = _start(tmp_path, "--every", "12", "--seconds", "10")
    yield base
    proc.kill()
    proc.wait()


@pytest.fixture
def keyed(tmp_path):
    proc, base = _start(tmp_path, "--key", "s3cret")
    yield base
    proc.kill()
    proc.wait()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def _code(url):
    try:
        return _get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


def test_health(bridge):
    code, headers, body = _get(bridge + "/health")
    h = json.loads(body)
    assert code == 200 and headers["Content-Type"] == "application/json"
    assert h["name"] == "test pump" and h["backend"] == "replay" and h["tickover_bridge"]
    assert {"level_dbfs", "last_clip_age_s", "uptime_s"} <= set(h)


def test_clip_length_and_format(bridge):
    for secs, want in (("7", 7), ("1", 3), ("999", 60)):
        code, headers, body = _get(f"{bridge}/clip?seconds={secs}")
        assert code == 200 and headers["Content-Type"] == "audio/wav"
        with wave.open(io.BytesIO(body)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)
            assert w.getnframes() == want * SR
    assert _code(bridge + "/clip?seconds=abc") == 400
    assert _code(bridge + "/clip?seconds=inf") == 400
    h = json.loads(_get(bridge + "/health")[2])
    assert h["level_dbfs"] is not None and -20 < h["level_dbfs"] < 0


def test_latest_with_timer(bridge):
    for _ in range(50):
        try:
            code, headers, body = _get(bridge + "/latest")
            break
        except urllib.error.HTTPError as e:
            assert e.code == 404
            time.sleep(0.1)
    assert code == 200 and "Last-Modified" in headers
    assert 0 <= float(headers["X-Tickover-Age"]) < 10
    with wave.open(io.BytesIO(body)) as w:
        assert w.getnframes() == 10 * SR   # timed clips are never shorter than a check needs


def test_latest_without_timer(keyed):
    try:
        _get(keyed + "/latest?key=s3cret")
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404 and "--every" in json.loads(e.read())["error"]


def test_key_required(keyed):
    assert _code(keyed + "/health") == 401
    assert _code(keyed + "/clip?seconds=3") == 401
    assert _code(keyed + "/health?key=wrong") == 401
    assert _code(keyed + "/health?key=s3cret") == 200
    assert _code(keyed + "/") == 401


def test_traversal_rejected(bridge):
    for path in ("/../../etc/passwd", "/clip/../../etc/passwd", "/%2e%2e/%2e%2e/windows/win.ini",
                 "/replay/a.wav", "/a.wav", "//etc/passwd"):
        assert _code(bridge + path) == 404, path
    req = urllib.request.Request(bridge + "/clip", method="POST", data=b"x")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 405
    assert _code(bridge + "/health?" + "&".join(f"a{i}=1" for i in range(12))) == 400


def test_open_network_needs_key_or_open(tmp_path):
    r = subprocess.run([sys.executable, BRIDGE, "--replay", str(tmp_path), "--host", "0.0.0.0", "--port", "1"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "--open" in r.stderr


def test_silence_is_refused(tmp_path):
    folder = tmp_path / "quiet"
    folder.mkdir()
    with wave.open(str(folder / "z.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\x00\x00" * SR * 4)
    port = _free_port()
    proc = subprocess.Popen([sys.executable, BRIDGE, "--replay", str(folder), "--host", "127.0.0.1", "--port", str(port),
                             "--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/clip?seconds=3", timeout=30)
        assert e.value.code == 503 and "silence" in json.loads(e.value.read())["error"]
        from tickover_core import audio
        with pytest.raises(ValueError, match="silence"):
            audio.load(f"http://127.0.0.1:{port}/clip?seconds=3&key=SECRET")
    finally:
        proc.kill()
        proc.wait()


def test_redact_url():
    from tickover_core import audio
    assert audio.redact_url("http://1.2.3.4:8765/clip?key=SECRET&seconds=10") == "http://1.2.3.4:8765/clip?key=***&seconds=10"
    assert "pw" not in audio.redact_url("rtsp://admin:pw@cam/stream")


def test_index_page(bridge):
    code, headers, body = _get(bridge + "/")
    assert code == 200 and b"/clip" in body and b"/latest" in body and b"/health" in body


def test_audio_load_from_bridge(bridge):
    from tickover_core import audio
    x = audio.load(bridge + "/clip?seconds=5")
    assert abs(len(x) - 5 * SR) <= SR // 100
    # verify asks for its own window: the bridge records exactly that
    y = audio.capture(bridge + "/clip", 4)
    assert abs(len(y) - 4 * SR) <= SR // 100
    info = audio.bridge_latest(bridge + "/latest")
    if info:  # the timer has written a clip by now
        assert info["stale"] is False and info["every_s"] == 12


def test_audio_load_without_ffmpeg(bridge, monkeypatch):
    from tickover_core import audio
    monkeypatch.setattr(audio, "ffmpeg", lambda: None)
    x = audio.load(bridge + "/clip?seconds=4")
    assert len(x) == 4 * SR


def test_device_not_answering():
    from tickover_core import audio
    url = f"http://127.0.0.1:{_free_port()}/clip?seconds=3"
    with pytest.raises(ValueError, match="is not answering"):
        audio.load(url)


def test_stale_bridge_is_reported(monkeypatch):
    from tickover_core import audio, tools
    monkeypatch.setattr(audio, "bridge_latest", lambda u: {"age_s": 7200.0, "every_s": 600.0, "limit_s": 1800.0,
                                                           "stale": True, "recorded_at": 1.0})
    r = json.loads(tools.check_tool({"machine": "bridge pump", "path": "http://127.0.0.1:9/latest"}))
    assert r["verdict"] == "stale" and "stopped recording" in r["message"]

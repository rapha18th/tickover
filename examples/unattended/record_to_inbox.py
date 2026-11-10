"""Keep one fresh recording on disk so Tickover can listen without anyone holding a phone.

Prefer bridge/tickover_bridge.py. It serves a fresh clip over HTTP, needs no shared folder,
runs on an old Android phone, a Raspberry Pi or any spare computer, and lets Hermes tell when
the device stops recording. See bridge/README.md. This script remains for setups where Hermes
can only read files.

Run this next to the machine: a Raspberry Pi with a USB microphone, an old laptop, a mini PC.
It records a short clip on a timer and replaces one file. Point a watch at that file:

    tickover_watch(machine="Borehole pump", schedule="every 2 hours",
                   source="/home/pi/tickover/pump.wav")

Hermes then checks the file on schedule and speaks only when the sound changed.

    pip install sounddevice
    python record_to_inbox.py /home/pi/tickover/pump.wav --every 600 --seconds 12

If the recorder and Hermes are on different computers, write into a folder both can see
(Syncthing, a network share, Dropbox). A phone works the same way: any recorder app that
saves into a synced folder is a source.

Needs only sounddevice and the standard library.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave

SR = 16000


def record(seconds: float, device) -> bytes:
    import sounddevice as sd

    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16", device=device)
    sd.wait()
    return x.tobytes()


def write(path: str, pcm: bytes) -> None:
    """Write beside the target, then swap. A reader never sees half a file."""
    tmp = path + ".part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record a short clip on a timer and keep one fresh file.")
    ap.add_argument("path", help="where the fresh recording lives, for example /home/pi/tickover/pump.wav")
    ap.add_argument("--every", type=float, default=600, help="seconds between recordings (default 600)")
    ap.add_argument("--seconds", type=float, default=12, help="length of each recording (default 12)")
    ap.add_argument("--device", help="microphone name or index; default is the system microphone")
    ap.add_argument("--once", action="store_true", help="record one clip and exit; use this from cron")
    a = ap.parse_args(argv)
    device = int(a.device) if a.device and a.device.isdigit() else a.device
    os.makedirs(os.path.dirname(os.path.abspath(a.path)), exist_ok=True)
    while True:
        try:
            write(a.path, record(a.seconds, device))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "wrote", a.path, flush=True)
        except Exception as exc:  # a pulled USB microphone should not kill the recorder
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "failed:", exc, file=sys.stderr, flush=True)
            if a.once:
                return 1
        if a.once:
            return 0
        time.sleep(max(1.0, a.every - a.seconds))


if __name__ == "__main__":
    sys.exit(main())

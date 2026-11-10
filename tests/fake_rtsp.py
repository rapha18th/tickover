"""A tiny RTSP camera for tests: OPTIONS, DESCRIBE, SETUP, PLAY over TCP, then L16 audio as interleaved RTP.

redirect_to makes every request answer 302 with that Location instead. raw(cseq) -> bytes makes every
request answer with exactly those bytes, for replies a real camera should never send.
"""
from __future__ import annotations

import math
import socket
import struct
import threading


class FakeCamera:
    def __init__(self, redirect_to: str | None = None, seconds: float = 4.0, raw=None):
        self.redirect_to, self.seconds, self.raw = redirect_to, seconds, raw
        self.requests: list[str] = []
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self):
        self.srv.close()

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    def _client(self, c):
        buf = b""
        try:
            while True:
                while b"\r\n\r\n" not in buf:
                    b = c.recv(4096)
                    if not b:
                        return
                    buf += b
                head, _, buf = buf.partition(b"\r\n\r\n")
                lines = head.decode(errors="replace").split("\r\n")
                self.requests.append(lines[0])
                method = lines[0].split(" ")[0]
                cseq = next((ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("cseq")), "1")
                if self.raw:
                    c.sendall(self.raw(cseq))
                    continue
                if self.redirect_to:
                    c.sendall(f"RTSP/1.0 302 Found\r\nCSeq: {cseq}\r\nLocation: {self.redirect_to}\r\n\r\n".encode())
                    continue
                if method == "OPTIONS":
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nPublic: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n\r\n".encode())
                elif method == "DESCRIBE":
                    sdp = ("v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=fake\r\nc=IN IP4 0.0.0.0\r\nt=0 0\r\n"
                           "m=audio 0 RTP/AVP 96\r\na=rtpmap:96 L16/16000/1\r\na=control:trackID=0\r\n")
                    c.sendall((f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\n"
                               f"Content-Length: {len(sdp)}\r\n\r\n{sdp}").encode())
                elif method == "SETUP":
                    c.sendall((f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 12345\r\n"
                               "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n").encode())
                elif method == "PLAY":
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 12345\r\n\r\n".encode())
                    self._stream(c)
                    return
                else:
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 12345\r\n\r\n".encode())
        except OSError:
            pass
        finally:
            c.close()

    def _stream(self, c):
        n, seq, ts = 320, 0, 0   # 20 ms packets
        for k in range(int(self.seconds * 50)):
            pcm = b"".join(struct.pack(">h", int(8000 * math.sin(2 * math.pi * 440 * (ts + i) / 16000))) for i in range(n))
            rtp = struct.pack(">BBHII", 0x80, 96, seq & 0xFFFF, ts & 0xFFFFFFFF, 0x1234) + pcm
            c.sendall(b"$\x00" + struct.pack(">H", len(rtp)) + rtp)
            seq, ts = seq + 1, ts + n

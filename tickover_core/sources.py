"""Which stream URLs Tickover may open, and how it opens them without being steered elsewhere.

A model can pass any URL to a tool. The owner decides which devices Tickover may reach.

The rule
    allowed_sources is a plugin setting: a list of host or host:port entries, empty by default. A URL
    (http, https, rtsp, rtsps) is accepted only when its host matches an entry and its port is the one
    the entry names. An entry without a port allows the scheme's default port (80, 443, 554, 322) and
    8765, the bridge port. Full URLs in the setting are reduced to host[:port].

    Tickover then resolves the host and checks every address it gets:
    - An entry that is an IP address (192.168.1.20, 127.0.0.1:8765, [fe80::1]) permits that address,
      whatever its class. The owner named it.
    - An entry that is a local name (pi.local, a single-word name, .lan, .home, .home.arpa, .internal,
      .localdomain, .ts.net, localhost) may resolve to private, loopback or carrier-grade NAT
      addresses. Those names live on the owner's network. Any other address it resolves to (link-local,
      reserved, a cloud metadata address) is dropped, and Tickover connects only to what is left.
    - Any other name must resolve to public addresses only. A public name cannot be pointed at the
      owner's network or a cloud metadata address later.
    - Link-local (169.254.0.0/16 with 169.254.169.254, fe80::/10) and reserved addresses pass only when
      the owner lists the address itself. Multicast and unspecified addresses are always refused.
    - A refusal for an address that is never a listening device (link-local, reserved, multicast,
      unspecified, a cloud metadata address) carries no command to allow it.
    IPv4-mapped, 6to4 and NAT64 IPv6 addresses are judged by the IPv4 address inside them.

Opening a URL
    Tickover connects to an address it checked, never to a second lookup (no DNS rebinding). It tries
    the checked addresses in order and uses the first that answers.
    http and https go through open_url(): it connects to the checked IP, sends the real Host header
    and the real name for TLS (SNI and certificate check), and follows at most three redirects by
    hand. Every hop runs the whole policy again: allowlist and resolved addresses. A redirect may only
    go to http or https.
    ffmpeg never opens an http(s) URL. Tickover reads the response through open_url() and pipes the
    bytes into ffmpeg on stdin, with -protocol_whitelist pipe. ffmpeg's own http client follows
    redirects and has no switch to stop that, so it is kept off the network.
    rtsp and rtsps go through RtspGuard: a relay on 127.0.0.1 that connects to a checked IP (TLS for
    rtsps, with the real name) and reads every reply the camera sends. ffmpeg follows an RTSP 3xx
    redirect to any host (tested with the bundled 7.1). So the relay reads each reply the way ffmpeg
    does and writes a new one from it: the status line, then only the headers a stream needs
    (Location is never among them), each line ending in CRLF. A 3xx reply, a REDIRECT request or a
    reply it cannot read drops the connection. ffmpeg runs with -protocol_whitelist rtsp,tcp and talks
    only to the relay.

Local paths
    On Windows a path that starts with two slashes or two backslashes (a UNC path, a device path) opens
    a network connection the moment Python or ffmpeg touches it. local_path() refuses those before any
    file call.

Outside Hermes (the CLI and tests) the list comes from TICKOVER_ALLOWED_SOURCES, a comma list.
"""
from __future__ import annotations

import base64
import http.client
import ipaddress
import os
import re
import socket
import ssl
import threading
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

SCHEMES = ("http", "https", "rtsp", "rtsps")
DEFAULT_PORTS = {"http": 80, "https": 443, "rtsp": 554, "rtsps": 322}
BRIDGE_PORT = 8765   # an entry without a port also allows the Tickover bridge's default port
# Cloud metadata services. Never a listening device: dropped for local names, and never offered an allow command.
METADATA = frozenset({"169.254.169.254", "169.254.170.2", "fd00:ec2::254", "100.100.100.200", "168.63.129.16"})
MAX_REDIRECTS = 3
SETTING = "allowed_sources"
ENV = "TICKOVER_ALLOWED_SOURCES"
LOCAL_SUFFIXES = (".local", ".lan", ".home", ".home.arpa", ".internal", ".localdomain", ".ts.net")
_BAD_CHARS = re.compile(r"[\s\\\x00-\x1f\x7f\"'`<>^{}|]")


class SourceRefused(ValueError):
    """The URL is not one the owner allowed, or it leads somewhere Tickover will not go."""


# ---------------------------------------------------------------- the setting
def _ctx():
    try:
        from . import agent
        return agent.CTX
    except Exception:  # noqa: BLE001
        return None


def plugin_id() -> str:
    return str(getattr(_ctx(), "plugin_id", "") or "tickover")


def raw_entries() -> list[str]:
    """allowed_sources as the owner wrote it: from the Hermes plugin settings, else the env var."""
    ctx = _ctx()
    get = getattr(ctx, "get_config", None)
    if callable(get):
        try:
            value = get(SETTING, [])
        except Exception:  # noqa: BLE001  an unreadable config allows nothing
            value = []
    else:
        value = os.environ.get(ENV, "")
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v or "").strip()]


def _host_port(text: str) -> tuple[str, int | None] | None:
    """('192.168.1.20', 8765) from '192.168.1.20:8765', 'http://192.168.1.20:8765/clip', '[::1]:8765' or 'pi.local'."""
    t = str(text or "").strip()
    if not t or _BAD_CHARS.search(t):
        return None
    if "://" in t:
        try:
            p = urlsplit(t)
            host, port = p.hostname, p.port
        except ValueError:
            return None
    elif t.startswith("["):
        m = re.fullmatch(r"\[([0-9a-fA-F:.]+)\](?::(\d{1,5}))?", t)
        if not m:
            return None
        host, port = m.group(1), int(m.group(2)) if m.group(2) else None
    elif t.count(":") > 1:   # a bare IPv6 address
        host, port = t, None
    else:
        host, _, p = t.partition(":")
        if p and not p.isdigit():
            return None
        port = int(p) if p else None
    host = _norm_host(host or "")
    if not host or (port is not None and not 0 < port < 65536):
        return None
    return host, port


def _norm_host(host: str) -> str:
    h = str(host).strip().lower().rstrip(".")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if "%" in h or not re.fullmatch(r"[a-z0-9.:_-]+", h or "-"):
        return ""
    ip = _ip(h)
    return str(ip) if ip else h


def entries() -> list[tuple[str, int | None]]:
    out = []
    for e in raw_entries():
        hp = _host_port(e)
        if hp and hp not in out:
            out.append(hp)
    return out


def _fmt(host: str, port: int | None) -> str:
    h = f"[{host}]" if ":" in host else host
    return f"{h}:{port}" if port else h


def allow_command(host: str, port: int | None) -> str:
    """The exact command the owner runs on the Hermes computer to allow this device, keeping the others.

    hermes config set reads a [a, b] value as a YAML list. The double quotes work in bash, PowerShell and cmd.
    """
    items = [_fmt(h, p) for h, p in entries()]
    new = _fmt(host, port)
    if new not in items:
        items.append(new)
    listed = ", ".join(f"'{i}'" if "[" in i or i.count(":") > 1 else i for i in items)   # IPv6 needs YAML quotes
    return f'hermes config set plugins.entries.{plugin_id()}.settings.{SETTING} "[{listed}]"'


# ---------------------------------------------------------------- addresses
def _ip(text: str):
    try:
        return ipaddress.ip_address(str(text).strip("[]"))
    except ValueError:
        return None


def _inner_v4(ip):
    """The IPv4 address an IPv6 address carries (mapped, 6to4, NAT64), else the address itself."""
    if ip.version == 6:
        if ip.ipv4_mapped:
            return ip.ipv4_mapped
        if ip.sixtofour:
            return ip.sixtofour
        if ip in ipaddress.ip_network("64:ff9b::/96"):
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip


_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_THIS_NET = ipaddress.ip_network("0.0.0.0/8")


def address_class(ip) -> str:
    """'public', 'private', 'loopback', 'cgnat', 'link-local', 'multicast', 'unspecified' or 'reserved'."""
    ip = _inner_v4(ip)
    if ip.is_unspecified or (ip.version == 4 and ip in _THIS_NET):
        return "unspecified"
    if ip.is_multicast:
        return "multicast"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.version == 4 and ip in _CGNAT:
        return "cgnat"
    if ip.version == 6 and ip.is_site_local:
        return "private"
    if ip.is_private and not ip.is_reserved:
        return "private"
    if ip.is_reserved or not ip.is_global:
        return "reserved"
    return "public"


def is_local_name(host: str) -> bool:
    h = host.lower().rstrip(".")
    return h == "localhost" or h.endswith(".localhost") or "." not in h or h.endswith(LOCAL_SUFFIXES)


def resolve(host: str, port: int) -> list[str]:
    ip = _ip(host)
    if ip:
        return [str(ip)]
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise LookupError(f"cannot find {host} on the network ({exc})") from exc
    out = []
    for info in infos:
        a = str(_ip(info[4][0].split("%")[0]) or "")
        if a and a not in out:
            out.append(a)
    if not out:
        raise LookupError(f"cannot find {host} on the network")
    return out


# ---------------------------------------------------------------- the check
@dataclass
class Target:
    url: str
    scheme: str
    host: str
    port: int
    addresses: list[str] = field(default_factory=list)

    @property
    def ip(self) -> str:
        return self.addresses[0]


def _redact(url: str) -> str:
    try:
        from .audio import redact_url
        return redact_url(url)
    except Exception:  # noqa: BLE001
        return str(url).split("?", 1)[0]


def parse(url: str) -> tuple[str, str, int]:
    """(scheme, host, port) of a URL Tickover can open. Raises SourceRefused for anything malformed."""
    u = str(url).strip()
    if _BAD_CHARS.search(u) or len(u) > 2048:
        raise SourceRefused(f"{_redact(u)} has characters a stream URL never needs. Give the plain URL.")
    try:
        p = urlsplit(u)
        port = p.port
    except ValueError as exc:
        raise SourceRefused(f"{_redact(u)} is not a valid URL ({exc}).") from exc
    scheme = p.scheme.lower()
    if scheme not in SCHEMES:
        raise SourceRefused(f"{_redact(u)}: Tickover opens http, https, rtsp and rtsps URLs only.")
    host = _norm_host(p.hostname or "")
    if not host:
        raise SourceRefused(f"{_redact(u)} has no usable host name.")
    return scheme, host, port or DEFAULT_PORTS[scheme]


def _entry_for(host: str, port: int, scheme: str) -> tuple[str, int | None] | None:
    """The entry that allows host:port. An entry without a port allows the scheme's default port and the bridge port."""
    for h, p in entries():
        if h == host and (p == port or (p is None and port in (DEFAULT_PORTS[scheme], BRIDGE_PORT))):
            return h, p
    return None


def _never_a_device(ip) -> bool:
    """A cloud metadata address, or a class no listening device has."""
    return (str(ip) in METADATA or str(_inner_v4(ip)) in METADATA
            or address_class(ip) in ("link-local", "reserved", "multicast", "unspecified"))


def check(url: str, resolve_host: bool = True) -> Target:
    """Apply the whole policy to one URL. Returns the Target to connect to. Raises SourceRefused.

    resolve_host=False checks the allowlist only (no DNS). A lookup failure raises LookupError.
    """
    scheme, host, port = parse(url)
    entry = _entry_for(host, port, scheme)
    lit = _ip(host)
    if entry is None and lit is not None and _never_a_device(lit):
        raise SourceRefused(f"{_fmt(host, port)} is not an allowed source. It is a {address_class(lit)} or cloud "
                            "metadata address, never a listening device.")
    if entry is None:
        listed = ", ".join(_fmt(h, p) for h, p in entries()) or "none yet"
        raise SourceRefused(
            f"{_fmt(host, port)} is not an allowed source. The owner allows each listening device once, on the "
            f"computer Hermes runs on: {allow_command(host, port)} . Allowed now: {listed}.")
    t = Target(url=str(url).strip(), scheme=scheme, host=host, port=port)
    if not resolve_host:
        return t
    found = resolve(host, port)
    literal = lit is not None
    local = not literal and is_local_name(host)
    kept = []
    for a in found:
        ip = ipaddress.ip_address(a)
        cls = address_class(ip)
        if cls in ("unspecified", "multicast"):
            raise SourceRefused(f"{_fmt(host, port)} resolves to {a}, a {cls} address. Tickover will not connect there.")
        if literal or cls == "public":
            kept.append(a)
        elif local:
            if cls in ("private", "loopback", "cgnat") and not _never_a_device(ip):
                kept.append(a)   # a local name lives on the owner's network; anything else it returns is dropped
        else:
            hint = ("List the address itself to allow it." if cls in ("link-local", "reserved") else
                    "Use the device's local name (pi.local) or list its IP address.")
            raise SourceRefused(f"{host} resolves to {a}, a {cls} address. Tickover refuses that for a public name, "
                                f"so it cannot be pointed into your network. {hint}")
    if not kept:
        raise SourceRefused(f"{host} is a local name, but it resolves only to {', '.join(found)}. For a local name "
                            "Tickover connects to private, loopback or carrier-grade NAT addresses only. List the "
                            "device's IP address instead.")
    t.addresses = kept
    return t


def _connect(addresses: list[str], port: int, timeout: float) -> socket.socket:
    """A connection to the first checked address that answers. Only checked addresses are ever tried."""
    err: OSError | None = None
    for a in addresses:
        try:
            return socket.create_connection((a, port), timeout)
        except OSError as exc:
            err = exc
    raise err or OSError("no address to connect to")


# ---------------------------------------------------------------- http(s), pinned


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, port, addresses, timeout):
        super().__init__(host, port, timeout=timeout)
        self._addresses = list(addresses)

    def connect(self):
        self.sock = _connect(self._addresses, self.port, self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, addresses, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._addresses = list(addresses)

    def connect(self):
        sock = _connect(self._addresses, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)   # SNI and the certificate use the name


class Response:
    """An open http(s) response from open_url(). read(n), status, headers, url, close()."""

    def __init__(self, conn, resp, url):
        self._conn, self._resp, self.url = conn, resp, url
        self.status, self.headers = resp.status, resp.headers

    def read(self, n: int = -1) -> bytes:
        return self._resp.read(n) if n is not None and n >= 0 else self._resp.read()

    def close(self) -> None:
        try:
            self._resp.close()
        finally:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_url(url: str, method: str = "GET", timeout: float = 15.0) -> Response:
    """Open an allowed http(s) URL at the address the policy checked, following at most three redirects.

    Each redirect target goes through check() again before Tickover connects to it, and it must be
    http or https. A 4xx or 5xx answer raises urllib.error.HTTPError, as urlopen would.
    """
    import urllib.error
    current = str(url).strip()
    auth_host = None
    for hop in range(MAX_REDIRECTS + 1):
        t = check(current)
        if t.scheme not in ("http", "https"):
            raise SourceRefused(f"{_redact(current)}: only http and https can be read here.")
        p = urlsplit(current)
        path = (p.path or "/") + (f"?{p.query}" if p.query else "")
        cls = _PinnedHTTPS if t.scheme == "https" else _PinnedHTTP
        conn = cls(t.host, t.port, t.addresses, timeout)
        headers = {"User-Agent": "tickover", "Accept": "*/*", "Connection": "close"}
        if p.username is not None and (auth_host in (None, t.host)):
            from urllib.parse import unquote
            cred = f"{unquote(p.username)}:{unquote(p.password or '')}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(cred).decode()
            auth_host = t.host
        try:
            conn.request(method, path, headers=headers)
            resp = conn.getresponse()
        except Exception:
            conn.close()
            raise
        if resp.status in (301, 302, 303, 307, 308) and resp.getheader("Location"):
            location = resp.getheader("Location")
            resp.close()
            conn.close()
            if hop == MAX_REDIRECTS:
                raise SourceRefused(f"{_redact(url)} redirected more than {MAX_REDIRECTS} times.")
            nxt = urljoin(current, location.strip())
            if urlsplit(nxt).scheme.lower() not in ("http", "https"):
                raise SourceRefused(f"{_redact(current)} redirected to a {urlsplit(nxt).scheme or 'relative'} URL. "
                                    "Tickover follows redirects to http and https only.")
            try:
                check(nxt)
            except SourceRefused as exc:
                raise SourceRefused(f"{_redact(current)} redirected to {_redact(nxt)}, which is refused: {exc}") from exc
            current = nxt
            continue
        if resp.status >= 400:
            raise urllib.error.HTTPError(current, resp.status, resp.reason, resp.headers, _Closing(resp, conn))
        return Response(conn, resp, current)
    raise SourceRefused(f"{_redact(url)} redirected more than {MAX_REDIRECTS} times.")


class _Closing:
    """File object for an HTTPError body that closes the connection with it."""

    def __init__(self, resp, conn):
        self._resp, self._conn = resp, conn

    def read(self, n=-1):
        return self._resp.read(n) if n is not None and n >= 0 else self._resp.read()

    def close(self):
        self._resp.close()
        self._conn.close()


# ---------------------------------------------------------------- rtsp(s), relayed
_MAX_HEAD = 65536
_MAX_BODY = 1_000_000
# Reply headers a stream needs. Everything else, Location first of all, is dropped.
_KEEP = {b"cseq": b"CSeq", b"session": b"Session", b"transport": b"Transport", b"content-type": b"Content-Type",
         b"content-base": b"Content-Base", b"public": b"Public", b"rtp-info": b"RTP-Info", b"range": b"Range",
         b"www-authenticate": b"WWW-Authenticate", b"authentication-info": b"Authentication-Info",
         b"server": b"Server", b"date": b"Date"}
_SERVER_REQUESTS = {b"OPTIONS", b"GET_PARAMETER", b"SET_PARAMETER", b"ANNOUNCE"}
_PRINTABLE = re.compile(rb"[^\x20-\x7e]")


class RtspRefused(OSError):
    """The camera sent a redirect, or a reply the relay will not pass on."""


def _atoi(word: bytes) -> int:
    """C atoi: the leading digits, 0 when there are none. ffmpeg reads the status code this way."""
    m = re.match(rb"\s*([+-]?\d+)", word)
    return int(m.group(1)) if m else 0


def rtsp_message(buf: bytes) -> tuple[bytes, int] | None:
    """Read one camera message from buf the way ffmpeg does, and write a clean one in its place.

    Returns (bytes to send to ffmpeg, bytes consumed), or None when buf holds less than a whole message.
    ffmpeg ends a line at LF, drops every CR, and ends the head at the first empty line. It reads the
    status code with atoi from the second word when the first word starts with RTSP/. The relay does
    the same, refuses a 3xx (ffmpeg would follow its Location) and a REDIRECT request, and sends a
    status line it wrote itself plus the headers in _KEEP, each ending in CRLF. Raises RtspRefused.
    """
    if buf[:1] == b"$":                                   # interleaved RTP or RTCP frame
        if len(buf) < 4:
            return None
        n = 4 + int.from_bytes(buf[2:4], "big")
        return (bytes(buf[:n]), n) if len(buf) >= n else None
    lines, pos = [], 0
    while True:
        nl = buf.find(b"\n", pos)
        if nl < 0:
            if len(buf) > _MAX_HEAD:
                raise RtspRefused("reply header too long")
            return None
        line = bytes(buf[pos:nl]).replace(b"\r", b"")
        pos = nl + 1
        if pos > _MAX_HEAD:
            raise RtspRefused("reply header too long")
        if not line:
            if not lines:                                 # blank lines between messages carry nothing
                return b"", pos
            break
        lines.append(line)
    words = lines[0].split()
    if words[0].startswith(b"RTSP/"):
        code = _atoi(words[1]) if len(words) > 1 else 0
        if 300 <= code < 400:
            raise RtspRefused("the camera answered with a redirect")
        if not 100 <= code < 600:
            raise RtspRefused("the camera sent a reply Tickover cannot read")
        reason = _PRINTABLE.sub(b"", b" ".join(words[2:]))[:80] or b"-"
        first = b"RTSP/1.0 %d %s" % (code, reason)
    else:
        method = words[0].upper()
        if method == b"REDIRECT":
            raise RtspRefused("the camera sent a redirect")
        if method not in _SERVER_REQUESTS or len(words) < 2:
            raise RtspRefused("the camera sent a request Tickover does not pass on")
        first = method + b" " + _PRINTABLE.sub(b"", words[1])[:512] + b" RTSP/1.0"
    out, length = [first], 0
    for line in lines[1:]:
        name, sep, value = line.partition(b":")
        key = name.strip().lower()
        if not sep or name[:1].isspace():
            continue                                      # folded or malformed lines are dropped
        if key == b"content-length":
            length = _atoi(value)
            if not 0 <= length <= _MAX_BODY:
                raise RtspRefused("the camera sent a reply Tickover cannot read")
            continue
        if key in _KEEP:
            out.append(_KEEP[key] + b": " + _PRINTABLE.sub(b"", value).strip())
    if len(buf) < pos + length:
        return None
    if length:
        out.append(b"Content-Length: %d" % length)
    return b"\r\n".join(out) + b"\r\n\r\n" + bytes(buf[pos:pos + length]), pos + length


class RtspGuard:
    """A relay on 127.0.0.1 between ffmpeg and one checked camera address.

    ffmpeg opens rtsp://127.0.0.1:<port>/<path>. Each connection goes to a checked IP (TLS with the
    real name for rtsps). Requests pass through unchanged. Replies go through rtsp_message() one by
    one: interleaved RTP frames pass, each RTSP reply is rewritten with the headers a stream needs, and
    a 3xx reply or a REDIRECT request closes the connection. So ffmpeg never learns another address.
    """

    def __init__(self, target: Target, timeout: float = 15.0):
        self.t, self.timeout = target, timeout
        self.refused: str | None = None
        self._socks: list[socket.socket] = []
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True, name="tickover-rtsp-guard").start()

    def local_url(self) -> str:
        p = urlsplit(self.t.url)
        user = ""
        if p.username is not None:
            user = p.username + (f":{p.password}" if p.password is not None else "") + "@"
        return f"rtsp://{user}127.0.0.1:{self.port}{p.path or '/'}" + (f"?{p.query}" if p.query else "")

    def _keep(self, s):
        with self._lock:
            self._socks.append(s)

    def _accept(self):
        while True:
            try:
                c, _ = self._srv.accept()
            except OSError:
                return
            self._keep(c)
            try:
                up = _connect(self.t.addresses, self.t.port, self.timeout)
                if self.t.scheme == "rtsps":
                    up = ssl.create_default_context().wrap_socket(up, server_hostname=self.t.host)
                up.settimeout(None)
            except OSError:
                c.close()
                continue
            self._keep(up)
            threading.Thread(target=self._pump, args=(c, up), daemon=True).start()
            threading.Thread(target=self._replies, args=(up, c), daemon=True).start()

    @staticmethod
    def _pump(src, dst):
        try:
            while True:
                b = src.recv(65536)
                if not b:
                    break
                dst.sendall(b)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass

    def _replies(self, up, down):
        buf = b""
        try:
            while True:
                b = up.recv(65536)
                if not b:
                    break
                buf += b
                while buf:
                    msg = rtsp_message(buf)
                    if msg is None:
                        break
                    data, n = msg
                    if data:
                        down.sendall(data)
                    buf = buf[n:]
        except RtspRefused as exc:
            self.refused = str(exc)
        except (OSError, ValueError):
            pass
        finally:
            for s in (up, down):
                try:
                    s.close()
                except OSError:
                    pass

    def close(self):
        try:
            self._srv.close()
        except OSError:
            pass
        with self._lock:
            for s in self._socks:
                try:
                    s.close()
                except OSError:
                    pass


# ---------------------------------------------------------------- local paths
def local_path(path) -> str:
    """The path, when opening it stays on this computer. Raises SourceRefused for a Windows network path.

    On Windows, \\\\host\\share and //host/share (and \\\\?\\UNC, \\\\.\\, \\??\\) make the system open an SMB
    or WebDAV connection as soon as anything checks the path, before ffmpeg runs. Nothing changes on
    Linux or macOS.
    """
    p = str(path).strip()
    if os.name != "nt":
        return p
    head = p.replace("/", "\\")
    refused = head.startswith(("\\\\", "\\??\\"))
    if not refused:
        drive = os.path.splitdrive(os.path.abspath(os.path.expanduser(p)))[0]
        refused = not re.fullmatch(r"[A-Za-z]:", drive)
    if refused:
        raise SourceRefused(f"{p} is a network path. Tickover reads files on this computer only. Copy the "
                            "recording here, or serve it from an allowed source.")
    return p

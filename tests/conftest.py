"""Shared test helpers: an empty source allowlist by default, and a fake DNS for names under test."""
import socket

import pytest


@pytest.fixture(autouse=True)
def _no_inherited_allowlist(monkeypatch):
    """Every test starts with no allowed sources, whatever the shell running pytest has set."""
    monkeypatch.delenv("TICKOVER_ALLOWED_SOURCES", raising=False)


@pytest.fixture
def allow(monkeypatch):
    """allow('127.0.0.1:8765', 'pi.local') sets the allowlist the way the CLI and tests read it."""
    def set_(*entries):
        monkeypatch.setenv("TICKOVER_ALLOWED_SOURCES", ",".join(entries))
    return set_


@pytest.fixture
def dns(monkeypatch):
    """dns({'cam.local': ['192.168.1.30']}) answers getaddrinfo for those names; other names resolve as usual."""
    real = socket.getaddrinfo
    table = {}

    def fake(host, port, *a, **kw):
        h = str(host).lower().rstrip(".")
        if h in table:
            out = []
            for ip in table[h]:
                fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
                addr = (ip, port or 0, 0, 0) if fam == socket.AF_INET6 else (ip, port or 0)
                out.append((fam, socket.SOCK_STREAM, 6, "", addr))
            return out
        return real(host, port, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", fake)

    def set_(mapping):
        table.update({k.lower(): list(v) for k, v in mapping.items()})
    return set_

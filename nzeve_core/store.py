"""Per-machine state under the plugin data dir: a memory bank (npz), calibration and a check log (SQLite)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np


def data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir  # type: ignore
        p = Path(plugin_data_dir("nzeve"))
    except Exception:  # outside Hermes (CLI, tests)
        p = Path(os.environ.get("NZEVE_HOME") or Path.home() / ".nzeve")
    p.mkdir(parents=True, exist_ok=True)
    return p


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError("machine name is empty")
    return s[:64]


def machine_dir(name: str) -> Path:
    p = data_dir() / "machines" / slug(name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_bank(name: str, bank: dict, meta: dict) -> None:
    d = machine_dir(name)
    np.savez_compressed(d / "bank.npz", **bank)
    (d / "machine.json").write_text(json.dumps(meta, indent=1))


def load_bank(name: str) -> tuple[dict, dict]:
    d = data_dir() / "machines" / slug(name)
    if not (d / "bank.npz").exists():
        raise KeyError(f"no machine named '{name}'. Enroll it first with nzeve_enroll.")
    z = np.load(d / "bank.npz")
    return {k: z[k] for k in z.files}, json.loads((d / "machine.json").read_text())


def machines() -> list[dict]:
    root = data_dir() / "machines"
    out = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        f = d / "machine.json"
        if f.exists():
            m = json.loads(f.read_text())
            m["last"] = last_checks(m["name"], 1)
            m["profile"] = profile(m["name"])
            out.append(m)
    return out


def _db() -> sqlite3.Connection:
    c = sqlite3.connect(data_dir() / "history.sqlite")
    c.execute("create table if not exists checks(ts real, machine text, score real, z real, verdict text, png text, src text)")
    return c


def log_check(machine: str, score: float, z: float, verdict: str, png: str, src: str, ts: float | None = None) -> None:
    with _db() as c:
        c.execute("insert into checks values(?,?,?,?,?,?,?)", (ts or time.time(), slug(machine), score, z, verdict, png, src))


def history(machine: str) -> list[dict]:
    with _db() as c:
        rows = c.execute("select ts, z, verdict, png, src from checks where machine=? order by ts", (slug(machine),)).fetchall()
    return [{"ts": r[0], "z": r[1], "verdict": r[2], "png": r[3], "src": r[4]} for r in rows]


# ---------------------------------------------------------------- maintenance profile
def profile(name: str) -> dict:
    f = data_dir() / "machines" / slug(name) / "profile.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_profile(name: str, **fields) -> dict:
    p = {**profile(name), **{k: v for k, v in fields.items() if v is not None}, "name": name}
    (machine_dir(name) / "profile.json").write_text(json.dumps(p, indent=1))
    return p


def last_checks(machine: str, n: int = 20) -> list[dict]:
    with _db() as c:
        rows = c.execute("select ts, z, verdict, png from checks where machine=? order by ts desc limit ?",
                         (slug(machine), n)).fetchall()
    return [{"ts": r[0], "z": r[1], "verdict": r[2], "png": r[3]} for r in rows]

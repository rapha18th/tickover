"""Hermes tools. Handlers return one JSON document; heavy imports happen inside handlers."""
from __future__ import annotations

import json
import os
import tempfile
import time


def _args(params) -> dict:
    if params is None:
        return {}
    if isinstance(params, str):
        params = json.loads(params or "{}")
    if not isinstance(params, dict):
        raise ValueError("tool arguments must be an object")
    return params


def _dump(payload) -> str:
    return json.dumps(payload, indent=1, default=str)


def _guard(fn):
    def run(params=None, **_kw):
        try:
            return _dump(fn(_args(params)))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            return _dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    run.__name__ = fn.__name__
    return run


def _paths(a) -> list[str]:
    p = a.get("paths") or a.get("path") or []
    p = [p] if isinstance(p, str) else list(p)
    missing = [x for x in p if not os.path.exists(x)]
    if missing:
        raise ValueError(f"file not found: {missing}")
    if not p:
        raise ValueError("give at least one audio or video file path")
    return p


@_guard
def enroll_tool(a):
    from . import engine
    return engine.enroll(str(a["machine"]), _paths(a), append=bool(a.get("append", False)))


@_guard
def check_tool(a):
    from . import engine
    r = engine.check(str(a["machine"]), _paths(a)[0])
    if r.get("card"):
        r["show_user"] = f"MEDIA:{r['card']}"
    return r


@_guard
def listen_tool(a):
    from . import audio, engine
    secs = float(a.get("seconds", 10))
    if not 3 <= secs <= 120:
        raise ValueError("seconds must be between 3 and 120")
    x = audio.record(secs, a.get("device"))
    import soundfile as sf
    path = os.path.join(tempfile.gettempdir(), f"earshot_{int(time.time())}.wav")
    sf.write(path, x, audio.SR)
    out = {"ok": True, "recorded": path, "seconds": secs, "level_db": round(audio.level_db(x), 1)}
    mode = a.get("then")
    if mode == "enroll":
        out["enroll"] = engine.enroll(str(a["machine"]), [path], append=True)
    elif mode == "check":
        out["check"] = engine.check(str(a["machine"]), path)
        out["show_user"] = f"MEDIA:{out['check'].get('card')}"
    return out


@_guard
def status_tool(a):
    from . import store
    ms = store.machines()
    if a.get("machine"):
        return {"ok": True, "machine": a["machine"], "history": store.last_checks(a["machine"], int(a.get("n", 20)))}
    return {"ok": True, "machines": [{"name": m["name"], "clips": m["n_clips"], "ready": m.get("ready"),
                                       "enrolled": time.strftime("%Y-%m-%d", time.localtime(m["enrolled_at"])),
                                       "last_check": m["last"][0] if m["last"] else None} for m in ms]}


def _schema(name, desc, props, req):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}


MACHINE = {"type": "string", "description": "A name for the machine, e.g. 'robot arm joint 3', 'generator', 'GPU fan'."}
FILES = {"type": "array", "items": {"type": "string"},
         "description": "Paths to audio or video files on this machine (wav, ogg/opus voice notes, m4a, mp4...)."}


def tool_specs() -> list[dict]:
    return [
        {"name": "earshot_enroll", "handler": enroll_tool, "emoji": "👂", "schema": _schema(
            "earshot_enroll",
            "Teach Earshot how one machine sounds when it is healthy. Give 3 or more recordings of about 10 s, "
            "or one recording of 30 s or more (it is split into 10 s clips). Use the same spot and distance each time. "
            "Runs locally; no audio leaves the machine.",
            {"machine": MACHINE, "paths": FILES,
             "append": {"type": "boolean", "description": "Add to an existing baseline instead of replacing it."}},
            ["machine", "paths"])},
        {"name": "earshot_check", "handler": check_tool, "emoji": "👂", "schema": _schema(
            "earshot_check",
            "Score a new recording against the machine's own healthy baseline. Returns drift in robust sigmas "
            "(under 2 normal, 2-3 watch, 3+ changed), which frequency bands moved, and a PNG card. "
            "Always show the card to the user by putting the returned show_user line (MEDIA:<path>) in your reply. "
            "Earshot detects change from the machine's own normal; it does not name the fault. Suggest what to inspect from the bands.",
            {"machine": MACHINE, "path": {"type": "string", "description": "Path to the recording to check."}},
            ["machine", "path"])},
        {"name": "earshot_listen", "handler": listen_tool, "emoji": "🎙️", "schema": _schema(
            "earshot_listen",
            "Record from this computer's microphone, then optionally enroll or check in one step. "
            "Use when the machine is next to the computer (a PC fan, a 3D printer, a robot on the bench).",
            {"seconds": {"type": "number", "description": "3-120 seconds. Default 10."},
             "machine": MACHINE,
             "then": {"type": "string", "enum": ["none", "enroll", "check"], "description": "What to do with the recording."},
             "device": {"type": "string", "description": "Optional input device name or index."}},
            [])},
        {"name": "earshot_status", "handler": status_tool, "emoji": "📈", "schema": _schema(
            "earshot_status",
            "List enrolled machines with their last check, or the check history of one machine.",
            {"machine": MACHINE, "n": {"type": "integer", "description": "History length. Default 20."}},
            [])},
    ]


def register_tools(ctx) -> None:
    for t in tool_specs():
        ctx.register_tool(name=t["name"], toolset="earshot", schema=t["schema"], handler=t["handler"],
                          description=t["schema"]["description"], emoji=t["emoji"])

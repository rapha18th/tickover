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


AUDIO_EXT = (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac", ".mp4", ".mov", ".webm", ".mkv")


def _expand(x: str) -> list[str]:
    """A path, a folder (all audio inside) or a glob pattern."""
    import glob
    x = os.path.expanduser(str(x).strip().strip('"'))
    if any(c in x for c in "*?["):
        return sorted(glob.glob(x))
    if os.path.isdir(x):
        return sorted(os.path.join(x, f) for f in os.listdir(x) if f.lower().endswith(AUDIO_EXT))
    return [x]


def _paths(a) -> list[str]:
    p = a.get("paths") or a.get("path") or []
    p = [p] if isinstance(p, str) else list(p)
    p = [y for x in p for y in _expand(x)]
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


def _when(v):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    import datetime as _dt
    return _dt.datetime.fromisoformat(str(v)).timestamp()


@_guard
def check_tool(a):
    from . import agent, engine
    m = str(a["machine"])
    r = engine.check(m, _paths(a)[0], recorded_at=_when(a.get("recorded_at")))
    agent.resolve(m)
    if r.get("verdict") == "healthy" and a.get("add_to_baseline"):
        r["baseline"] = engine.enroll(m, _paths(a)[:1], append=True)["message"]
    if r.get("card"):
        r["show_user"] = f"MEDIA:{r['card']}"
    return r


@_guard
def watch_tool(a):
    from . import agent
    return agent.watch(str(a["machine"]), a.get("kind", "other"), a.get("location"),
                       a.get("schedule", "every monday 9am"), a.get("tip"))


@_guard
def checkin_tool(a):
    from . import agent
    return agent.checkin(str(a["machine"]))


@_guard
def fleet_tool(a):
    from . import agent
    return agent.fleet()


@_guard
def listen_tool(a):
    from . import audio, engine
    secs = float(a.get("seconds", 10))
    if not 3 <= secs <= 120:
        raise ValueError("seconds must be between 3 and 120")
    x = audio.record(secs, a.get("device"))
    import soundfile as sf
    path = os.path.join(tempfile.gettempdir(), f"nzeve_{int(time.time())}.wav")
    sf.write(path, x, audio.SR)
    out = {"ok": True, "recorded": path, "seconds": secs, "level_db": round(audio.level_db(x), 1)}
    mode = a.get("then")
    if mode == "enroll":
        out["enroll"] = engine.enroll(str(a["machine"]), [path], append=True)
    elif mode == "check":
        out["check"] = engine.check(str(a["machine"]), path)
        out["show_user"] = f"MEDIA:{out['check'].get('card')}"
    return out


def _schema(name, desc, props, req):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}


MACHINE = {"type": "string", "description": "A name for the machine, e.g. 'robot arm joint 3', 'generator', 'GPU fan'."}
FILES = {"type": "array", "items": {"type": "string"},
         "description": "Audio or video files on this machine (wav, ogg/opus voice notes, m4a, mp4...). Each entry may also be a folder or a glob such as C:/recordings/healthy_*.wav."}


def tool_specs() -> list[dict]:
    return [
        {"name": "nzeve_enroll", "handler": enroll_tool, "emoji": "👂", "schema": _schema(
            "nzeve_enroll",
            "Teach Nzeve how one machine sounds when it is healthy. Give 3 or more recordings of about 10 s, "
            "or one recording of 30 s or more (it is split into 10 s clips). Use the same spot and distance each time. "
            "Runs locally; no audio leaves the machine.",
            {"machine": MACHINE, "paths": FILES,
             "append": {"type": "boolean", "description": "Add to an existing baseline instead of replacing it."}},
            ["machine", "paths"])},
        {"name": "nzeve_check", "handler": check_tool, "emoji": "👂", "schema": _schema(
            "nzeve_check",
            "Score a new recording against the machine's own healthy baseline. Returns drift in robust sigmas "
            "(under 2 normal, 2-3 watch, 3+ changed), which frequency bands moved, the trend over past check-ins "
            "(slope per week, days until the drift line, whether to escalate) and a PNG card. "
            "Always show the card by putting the returned show_user line (MEDIA:<path>) on its own line in your reply. "
            "Nzeve detects change from the machine's own normal; it does not name the fault. Suggest what to inspect "
            "from the bands and the kind of machine, and say it is a guess.",
            {"machine": MACHINE, "path": {"type": "string", "description": "Path to the recording to check (a folder or glob works too)."},
             "recorded_at": {"type": "string", "description": "Optional ISO date the recording was made, when it is not today."},
             "add_to_baseline": {"type": "boolean", "description": "If the owner confirms the machine is fine and the verdict is healthy, add this recording to the baseline."}},
            ["machine", "path"])},
        {"name": "nzeve_watch", "handler": watch_tool, "emoji": "🗓️", "schema": _schema(
            "nzeve_watch",
            "Put a machine on a listening routine: save what it is and where, how to record it, and schedule a "
            "Hermes cron check-in (default every Monday 9am) that asks the owner, in this chat, for a 10-second recording. "
            "Use for machines with no sensors: generators, borehole and water pumps, compressors, fans, older vehicles.",
            {"machine": MACHINE,
             "kind": {"type": "string", "description": "genset, pump, fan, vehicle, compressor, printer, robot or other."},
             "location": {"type": "string", "description": "Where it is, e.g. 'clinic backyard, Gweru'."},
             "schedule": {"type": "string", "description": "Hermes schedule, e.g. 'every monday 9am', 'every 3 days', '0 8 * * 1'. Empty for no reminders."},
             "tip": {"type": "string", "description": "How to record this machine, if the default for its kind does not fit."}},
            ["machine"])},
        {"name": "nzeve_checkin", "handler": checkin_tool, "emoji": "📨", "schema": _schema(
            "nzeve_checkin",
            "Mark that a recording of this machine is expected from its owner now, and get the recording instructions to "
            "send them. The scheduled check-in job calls this; the next voice note or audio file from that chat is routed "
            "to nzeve_check automatically.",
            {"machine": MACHINE}, ["machine"])},
        {"name": "nzeve_fleet", "handler": fleet_tool, "emoji": "📋", "schema": _schema(
            "nzeve_fleet",
            "Report on every watched machine: state (steady, worsening, watch, drift), drift trend per week, forecast "
            "days until the drift line, overdue check-ins, and one fleet card image (show it with its MEDIA: line).",
            {}, [])},
        {"name": "nzeve_listen", "handler": listen_tool, "emoji": "🎙️", "schema": _schema(
            "nzeve_listen",
            "Record from this computer's microphone, then optionally enroll or check in one step. "
            "Use when the machine is next to the computer (a PC fan, a 3D printer, a robot on the bench).",
            {"seconds": {"type": "number", "description": "3-120 seconds. Default 10."},
             "machine": MACHINE,
             "then": {"type": "string", "enum": ["none", "enroll", "check"], "description": "What to do with the recording."},
             "device": {"type": "string", "description": "Optional input device name or index."}},
            [])},
    ]


def register_tools(ctx) -> None:
    from . import agent
    agent.CTX = ctx
    for t in tool_specs():
        ctx.register_tool(name=t["name"], toolset="nzeve", schema=t["schema"], handler=t["handler"],
                          description=t["schema"]["description"], emoji=t["emoji"])
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_gateway_dispatch", agent.on_inbound)
        ctx.register_hook("pre_llm_call", agent.on_turn)

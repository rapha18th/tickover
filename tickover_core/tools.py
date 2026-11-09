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
        except Exception as exc:  # noqa: BLE001  a tool must answer with JSON, whatever went wrong
            return _dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    run.__name__ = fn.__name__
    return run


AUDIO_EXT = (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac", ".mp4", ".mov", ".webm", ".mkv", ".3gp", ".amr")


def _is_url(x) -> bool:
    from . import audio
    return audio.is_url(x)


def _expand(x: str) -> list[str]:
    """A path, a URL, a folder (all audio inside) or a glob pattern."""
    import glob
    x = str(x).strip().strip('"')
    if _is_url(x):
        return [x]
    x = os.path.expanduser(x)
    if os.path.isfile(x):  # a real file wins, even with [ or ? in its name
        return [x]
    if any(c in x for c in "*?["):
        return sorted(glob.glob(x))
    if os.path.isdir(x):
        return sorted(os.path.join(x, f) for f in os.listdir(x) if f.lower().endswith(AUDIO_EXT))
    return [x]


def _paths(a) -> list[str]:
    p = a.get("paths") or a.get("path") or []
    p = [p] if isinstance(p, str) else list(p)
    p = list(dict.fromkeys(y for x in p for y in _expand(x)))
    missing = [x for x in p if not _is_url(x) and not os.path.exists(x)]
    if missing:
        raise ValueError(f"file not found: {missing}")
    if not p:
        raise ValueError("give at least one audio or video file path, or a stream URL")
    return p


def _newest(p: list[str]) -> str:
    """One recording to score. A folder or pattern means the latest file in it."""
    files = [x for x in p if not _is_url(x)]
    if len(p) == 1 or len(files) != len(p):
        return p[0]
    return max(files, key=os.path.getmtime)


def _latest(a) -> str:
    return _newest(_paths(a))


def _source(a) -> str | None:
    s = a.get("source") or a.get("path")
    if not s:
        return None
    s = _expand(s)
    if not s:
        raise ValueError("source matched no file")
    s = [_newest(s)]
    if not _is_url(s[0]) and not os.path.exists(s[0]):
        raise ValueError(f"file not found: {s[0]}")
    return s[0]


def _show(r: dict) -> dict:
    if r.get("card"):
        r["show_user"] = f"MEDIA:{r['card']}"
    return r


def _when(v):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    import datetime as _dt
    return _dt.datetime.fromisoformat(str(v)).timestamp()


@_guard
def hear_tool(a):
    from . import hear
    top = int(a.get("top", 5))
    if not 1 <= top <= 15:
        raise ValueError("top must be between 1 and 15")
    return _show(hear.hear(_latest(a), top=top))


@_guard
def enroll_tool(a):
    from . import engine
    return engine.enroll(str(a["machine"]), _paths(a), append=bool(a.get("append", False)),
                         ignore_quality=bool(a.get("ignore_quality", False)))


def _unattended(m: str, path: str):
    """A watched file must be new. Returns (early result or None, mtime or None)."""
    import os
    import time
    from . import store
    prof = store.profile(m)
    src = str(prof.get("source") or "")
    if _is_url(path) or not src or _is_url(src):
        return None, None
    watched = {os.path.realpath(p) for p in _expand(src)}  # the file itself, or the files in a watched folder
    if os.path.realpath(path) not in watched:
        return None, None
    mtime = os.path.getmtime(path)
    age = time.time() - mtime
    limit = max(2 * float(prof.get("every_days") or 7.0) * 86400, 3600)
    if age > limit:
        return {"ok": True, "machine": m, "verdict": "stale", "logged": False, "age_hours": round(age / 3600, 1),
                "message": f"The recording at {path} is {age / 3600:.0f} hours old. The recorder has stopped writing. "
                           "Nothing was scored. Tell the owner the microphone or recorder needs attention."}, mtime
    if prof.get("source_mtime") == mtime:
        return {"ok": True, "machine": m, "verdict": "same_recording", "logged": False,
                "message": "This is the recording that was checked last time. Nothing new to hear yet."}, mtime
    return None, mtime


def _check(m: str, path: str, a: dict) -> dict:
    from . import agent, engine
    early, mtime = _unattended(m, path)
    if early:
        return early
    r = engine.check(m, path, recorded_at=_when(a.get("recorded_at")), ignore_quality=bool(a.get("ignore_quality", False)))
    if r.get("verdict") in ("retake", "insufficient"):
        return r   # nothing was heard about the machine: the request for a recording stays open
    agent.resolve(m)
    if mtime is not None:
        from . import store
        store.save_profile(m, source_mtime=mtime)
    if r.get("verdict") == "healthy" and a.get("add_to_baseline"):
        r["baseline"] = engine.enroll(m, [path], append=True)["message"]
    _show(r)
    from . import teach
    dets = teach.active_for(m)
    if dets:
        r["detectors"] = dets
        r["next"] = "Also call tickover_teach(action='classify', detector=<name>, path=<same path>) for each detector."
    return r


@_guard
def check_tool(a):
    return _check(str(a["machine"]), _latest(a), a)


@_guard
def verify_tool(a):
    from . import verify
    step = str(a.get("step", "after")).lower()
    secs = float(a.get("seconds", 6))
    if not 3 <= secs <= 30:
        raise ValueError("seconds must be between 3 and 30")
    src = _source(a)
    if step == "before":
        return verify.before(src, secs, a.get("label"), a.get("device"))
    if step == "after":
        bp = a.get("before_path")
        if bp and not os.path.exists(bp):
            raise ValueError(f"file not found: {bp}")
        return _show(verify.after(src, secs, a.get("expect", "change"), a.get("label"), a.get("machine"), bp, a.get("device")))
    if step == "clip":
        if not src:
            raise ValueError("step='clip' needs source: one recording with the action inside it")
        return _show(verify.clip(src, a.get("expect", "change"), a.get("label"), a.get("machine")))
    raise ValueError("step must be before, after or clip")


@_guard
def teach_tool(a):
    from . import teach
    act = a.get("action", "add")
    if act == "list":
        return teach.listing(a.get("machine"))
    name = str(a["detector"])
    if act == "add":
        return teach.add(name, str(a["label"]), _paths(a), a.get("machine"))
    if act == "test":
        return teach.test(name)
    if act == "classify":
        return teach.classify(name, _latest(a))
    if act == "delete":
        return teach.delete(name)
    raise ValueError("action must be add, test, classify, list or delete")


@_guard
def watch_tool(a):
    from . import agent
    src = a.get("source")
    if src and not _is_url(src):
        src = os.path.abspath(os.path.expanduser(str(src)))
    return agent.watch(str(a["machine"]), a.get("kind", "other"), a.get("location"),
                       a.get("schedule", "every monday 9am"), a.get("tip"), src or None)


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
    mode = a.get("then") or "hear"
    if mode in ("enroll", "check") and not a.get("machine"):
        raise ValueError(f"then='{mode}' needs machine")
    x = audio.record(secs, a.get("device"))
    import soundfile as sf
    path = os.path.join(tempfile.gettempdir(), f"tickover_{int(time.time() * 1000)}.wav")
    sf.write(path, x, audio.SR)
    out = {"ok": True, "recorded": path, "seconds": secs, "level_db": round(audio.level_db(x), 1)}
    if mode == "enroll":
        out["enroll"] = engine.enroll(str(a["machine"]), [path], append=True)
    elif mode == "check":
        out["check"] = _check(str(a["machine"]), path, a)
        if out["check"].get("card"):
            out["show_user"] = f"MEDIA:{out['check']['card']}"
    elif mode == "hear":
        from . import hear
        out["hear"] = hear.hear(path)
        if out["hear"].get("card"):
            out["show_user"] = f"MEDIA:{out['hear']['card']}"
    return out


@_guard
def selftest_tool(a):
    from . import selftest
    return _show(selftest.run(verbose=False))


def _schema(name, desc, props, req):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}


MACHINE = {"type": "string", "description": "A name for the machine, e.g. 'robot arm joint 3', 'generator', 'GPU fan'."}
FILES = {"type": "array", "items": {"type": "string"},
         "description": "Audio or video files on this computer (wav, ogg/opus voice notes, m4a, mp4...). Each entry may also be a "
                        "folder, a glob such as C:/recordings/healthy_*.wav, or a stream URL (http, https, rtsp)."}
SOURCE = ("A file path, or a stream URL (rtsp:// camera, http:// audio stream). Leave it out to use this computer's microphone.")
IGNORE_Q = {"type": "boolean", "description": "Score it even though the quality gate objects. Only when the owner says the "
                                              "recording is as good as it gets, e.g. a workshop with a radio that is always on."}


def tool_specs() -> list[dict]:
    return [
        {"name": "tickover_hear", "handler": hear_tool, "emoji": "👂", "schema": _schema(
            "tickover_hear",
            "Listen to any recording. Use this first on any audio that is not speech to you: a voice note of a machine, "
            "a video, a camera stream. Returns what it sounds like (the top AudioSet tags with 0-1 scores and the seconds "
            "where each is strongest), whether it sounds like a machine, whether the recording is clean enough to use "
            "(verdict 'retake' with the fix when it is clipped, silent, too short, or has speech or wind over it), and a "
            "PNG card. Needs no setup and no baseline. Runs locally. Tags are what a general audio model hears: say "
            "'sounds like', never 'is', and never name a fault from them.",
            {"path": {"type": "string", "description": "The recording: a file path or a stream URL."},
             "top": {"type": "integer", "description": "How many tags to return. Default 5."}},
            ["path"])},
        {"name": "tickover_verify", "handler": verify_tool, "emoji": "✅", "schema": _schema(
            "tickover_verify",
            "Hear whether an action took effect. Use it after you switch something on or off with any other tool (a smart "
            "plug, a relay, a shell command, a robot move) and nothing reports back: did the pump start, did the fan stop, "
            "did the printer begin. Three steps. step='before': capture a few seconds before the action. Then run the "
            "action and wait a few seconds for the machine to settle. step='after' with expect='start', 'stop' or "
            "'change': capture again and compare. Or step='clip' with one recording that has the action inside it; it "
            "finds the moment; it needs 9 s or more with at least 4 s on each side of the action. Returns verdict 'confirmed', 'no_change', 'other_change' or 'unclear' (the change sits too close to the edge of the clip), how far the tone and the level "
            "moved in sigmas of the sound's own variation, what appeared or faded, and a PNG card. Needs no baseline. "
            "Verify hears that the sound changed. It does not prove what changed, and it does not detect faults: use "
            "tickover_check for that. On no_change, say the action may not have reached the machine or the microphone "
            "may not hear it; do not claim success. Before and after must come from the same microphone in the same place.",
            {"step": {"type": "string", "enum": ["before", "after", "clip"]},
             "source": {"type": "string", "description": SOURCE + " Use the same source for before and after. Required for clip."},
             "seconds": {"type": "number", "description": "Length of the window from a microphone or stream, 3-30. Default 6."},
             "expect": {"type": "string", "enum": ["start", "stop", "change"],
                        "description": "start: something should now be running. stop: it should be gone. change: any change."},
             "label": {"type": "string", "description": "What the action was, for the card: 'borehole pump on'."},
             "machine": {"type": "string", "description": "Optional enrolled machine. After a start, also says whether it sounds like its healthy self."},
             "before_path": {"type": "string", "description": "For after: a before recording on disk, instead of a stored before step."},
             "device": {"type": "string", "description": "Optional microphone name or index."}},
            ["step"])},
        {"name": "tickover_enroll", "handler": enroll_tool, "emoji": "👂", "schema": _schema(
            "tickover_enroll",
            "Remember how one machine sounds when it is healthy. Give 3 or more recordings of about 10 s, "
            "or one recording of 30 s or more (it is split into 10 s clips). Use the same spot and distance each time. "
            "Recordings that fail the quality gate are left out and listed in 'skipped' with the fix; pass that on. "
            "Runs locally; no audio leaves this computer.",
            {"machine": MACHINE, "paths": FILES,
             "append": {"type": "boolean", "description": "Add to an existing baseline instead of replacing it."},
             "ignore_quality": IGNORE_Q},
            ["machine", "paths"])},
        {"name": "tickover_check", "handler": check_tool, "emoji": "🩺", "schema": _schema(
            "tickover_check",
            "Score a new recording against the machine's own healthy baseline. Returns drift in robust sigmas "
            "(under 2 normal, 2-3 watch, 3+ changed), which frequency bands moved, the trend over past check-ins "
            "(slope per week, days until the drift line, whether to escalate) and a PNG card. "
            "Always show the card by putting the returned show_user line (MEDIA:<path>) on its own line in your reply. "
            "Verdict 'retake' means the recording was not usable and nothing was scored or logged: pass on the fix and ask "
            "for another; it says nothing about the machine. A first 'drift' comes with confirmed=false: ask for a second "
            "recording before telling the owner to act. Tickover detects change from the machine's own normal; it does "
            "not name the fault. Suggest what to inspect from the bands and the kind of machine, and say it is a guess.",
            {"machine": MACHINE, "path": {"type": "string", "description": "The recording to check: a file path, a folder or glob, or a stream URL."},
             "recorded_at": {"type": "string", "description": "Optional ISO date the recording was made, when it is not today."},
             "add_to_baseline": {"type": "boolean", "description": "If the owner confirms the machine is fine and the verdict is healthy, add this recording to the baseline."},
             "ignore_quality": IGNORE_Q},
            ["machine", "path"])},
        {"name": "tickover_teach", "handler": teach_tool, "emoji": "🧠", "schema": _schema(
            "tickover_teach",
            "Teach Tickover a new question about one machine, in the owner's words: 'is the pump running dry?', "
            "'loaded or idling?'. Turn the question into two or more named states (labels), ask for 5 separate recordings "
            "of about 10 s for each state from the same spot (4 is the minimum), and add them (action=add). It then tests "
            "itself by holding out one recording at a time and reports n of N right next to what guessing would get. It "
            "switches on only when that result could not be luck; otherwise it says why and stays off. Then classify new "
            "recordings (action=classify): the closest state, 'uncertain' when two states are too close to call, or "
            "'unknown' when it matches none, 'not_ready' while it is switched off. No training, runs locally. Expect it to stay off for hard questions. Report counts, not percentages. A label is the "
            "owner's name for a state, not a diagnosis.",
            {"action": {"type": "string", "enum": ["add", "test", "classify", "list", "delete"]},
             "detector": {"type": "string", "description": "Name for this question, e.g. 'borehole pump dry run'."},
             "label": {"type": "string", "description": "For add: the state these recordings show, e.g. 'normal', 'running dry'."},
             "paths": FILES,
             "path": {"type": "string", "description": "For classify: the recording to label."},
             "machine": {"type": "string", "description": "The watched machine this detector belongs to; its check-ins then suggest running it."}},
            ["action"])},
        {"name": "tickover_watch", "handler": watch_tool, "emoji": "🗓️", "schema": _schema(
            "tickover_watch",
            "Put a machine on a listening routine with a Hermes cron job (default every Monday 9am). Two modes. With no "
            "source, the job asks the owner in this chat for a 10-second recording, and their next voice note is checked "
            "automatically: for machines with no sensors, such as generators, borehole pumps, compressors, older vehicles. "
            "With a source (a camera or stream URL, or a file a device keeps fresh), the job listens by itself and messages "
            "the owner only when the sound has changed or the microphone needs attention. Calling it again for the same "
            "machine replaces the schedule.",
            {"machine": MACHINE,
             "kind": {"type": "string", "description": "genset, pump, fan, vehicle, compressor, printer, robot or other."},
             "location": {"type": "string", "description": "Where it is, e.g. 'clinic backyard, Gweru'."},
             "schedule": {"type": "string", "description": "Hermes schedule, e.g. 'every monday 9am', 'every 6 hours', '0 8 * * 1'. Empty for no schedule."},
             "tip": {"type": "string", "description": "How to record this machine, if the default for its kind does not fit."},
             "source": {"type": "string", "description": "Unattended mode: an rtsp:// or http:// stream, or a file path a recorder overwrites."}},
            ["machine"])},
        {"name": "tickover_checkin", "handler": checkin_tool, "emoji": "📨", "schema": _schema(
            "tickover_checkin",
            "Mark that a recording of this machine is expected from its owner now, and get the recording instructions to "
            "send them. The scheduled check-in job calls this; the next voice note or audio file from that chat is routed "
            "to tickover_check automatically.",
            {"machine": MACHINE}, ["machine"])},
        {"name": "tickover_fleet", "handler": fleet_tool, "emoji": "📋", "schema": _schema(
            "tickover_fleet",
            "Report on every watched machine: state (steady, worsening, watch, drift), drift trend per week, forecast "
            "days until the drift line, overdue check-ins, and one fleet card image (show it with its MEDIA: line).",
            {}, [])},
        {"name": "tickover_listen", "handler": listen_tool, "emoji": "🎙️", "schema": _schema(
            "tickover_listen",
            "Record from this computer's microphone, then hear, enroll or check in one step. "
            "Use when the machine is next to the computer (a PC fan, a 3D printer, a robot on the bench).",
            {"seconds": {"type": "number", "description": "3-120 seconds. Default 10."},
             "machine": MACHINE,
             "then": {"type": "string", "enum": ["hear", "enroll", "check", "none"], "description": "What to do with the recording. Default hear."},
             "device": {"type": "string", "description": "Optional input device name or index."}},
            [])},
        {"name": "tickover_selftest", "handler": selftest_tool, "emoji": "🧪", "schema": _schema(
            "tickover_selftest",
            "Prove that Tickover works on this computer, in about a minute, with no recordings: it synthesises a fan, "
            "enrolls it, checks a healthy and a faulty clip, verifies a start, and reports what is installed (models, "
            "ffmpeg, microphone). Run it once after install, or when something looks wrong. The first run downloads the 87 MB listening model once; after that nothing leaves this computer. It leaves nothing else behind.",
            {}, [])},
    ]


def register_tools(ctx) -> None:
    from . import agent
    agent.CTX = ctx
    for t in tool_specs():
        ctx.register_tool(name=t["name"], toolset="tickover", schema=t["schema"], handler=t["handler"],
                          description=t["schema"]["description"], emoji=t["emoji"])
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_gateway_dispatch", agent.on_inbound)
        ctx.register_hook("pre_llm_call", agent.on_turn)

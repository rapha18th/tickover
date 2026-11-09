"""The routine: watch a machine, ask for a recording on a schedule, catch the reply.

watch()    saves the machine's profile (kind, place, how to record it) and schedules a
           Hermes cron job. With no source the job asks the owner for a recording, in the
           chat it was set up from. With a source (a camera, a stream, a file a device keeps
           fresh) the job listens by itself and speaks only when something changed.
checkin()  is what the asking job calls: it marks a recording as expected from that chat and
           returns the request in plain words.
Hooks      pre_gateway_dispatch notes any audio that arrives in a chat (speech-to-text drops
           the file path when machine noise transcribes to nothing). pre_llm_call hands that
           path to the agent: as the expected recording when one is due, otherwise as "this
           sounds like a machine" when it holds no speech. A spoken voice message is left
           alone.
fleet()    renders every machine on one card with its trend and forecast.
"""
from __future__ import annotations

import json
import os
import re
import time

from . import store, trend

CTX = None  # the Hermes plugin context, set at register()
PENDING_TTL = 72 * 3600
INBOX_TTL = 600
SPEECH_IS_TALK = 0.5   # CED speech probability above which a voice note is someone talking to the agent
AUDIO_EXT = (".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac", ".wav", ".flac", ".mp4", ".mov", ".webm", ".3gp", ".amr")

TIPS = {
    "genset": "engine warm, at idle, phone 30 cm from the engine block, same side as last time",
    "pump": "pump running, phone resting on the same spot of the pump housing or pipe",
    "fan": "fan at its usual speed, phone 30 cm in front of the hub",
    "vehicle": "engine warm, at idle, bonnet open, phone above the engine at the same spot",
    "compressor": "compressor running under load, phone 30 cm from the head",
    "printer": "a print running, phone on the frame next to the motors",
    "robot": "the same motion cycle, phone at the same spot near the joint",
}


def _session(name: str) -> str | None:
    try:
        from gateway.session_context import get_session_env  # public Hermes surface
        return get_session_env(name) or None
    except Exception:  # noqa: BLE001
        return os.environ.get(name) or None


def _chat_key(platform, chat_id) -> str | None:
    return f"{platform}:{chat_id}" if platform and chat_id else None


def _state_path():
    return store.data_dir() / "routine.json"


def _load() -> dict:
    p = _state_path()
    try:
        s = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        s = {}
    s.setdefault("pending", {})
    s.setdefault("inbox", {})
    return s


def _save(s: dict) -> None:
    _state_path().write_text(json.dumps(s, indent=1))


_UNIT_DAYS = {"m": 1 / 1440, "min": 1 / 1440, "h": 1 / 24, "hour": 1 / 24, "d": 1.0, "day": 1.0, "w": 7.0, "week": 7.0}
_DOW = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def every_days(schedule: str | None) -> float:
    """Days between runs of a Hermes schedule string. Used to tell when a check-in is overdue."""
    s = (schedule or "").strip().lower()
    if not s:
        return 7.0
    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s*(minute|min|hour|day|week|m|h|d|w)", s)
    if m:
        return max(float(m.group(1)) * _UNIT_DAYS[m.group(2)], 1 / 1440)
    m = re.fullmatch(r"(\d+)\s*(m|h|d|w)", s)
    if m:
        return float(m.group(1)) * _UNIT_DAYS[m.group(2)]
    if "hourly" in s or "every hour" in s:
        return 1 / 24
    if "daily" in s or "every day" in s or "weekday" in s:
        return 1.0
    f = s.split()
    if len(f) == 5 and all(re.fullmatch(r"[\d*/,\-a-z]+", x) for x in f):   # cron: minute hour dom month dow
        minute, hour, dom, _month, dow = f
        if dow != "*":
            return 7.0 / max(1, len(dow.split(",")))
        if dom != "*":
            return 30.0
        if hour == "*":
            return 1 / 24
        m = re.fullmatch(r"\*/(\d+)", hour)
        return int(m.group(1)) / 24 if m else 1.0 / max(1, len(hour.split(",")))
    days = sum(d in s for d in _DOW)
    return 7.0 / days if days else 7.0


# ---------------------------------------------------------------- tools
def _ask_prompt(machine, kind, location, tip) -> str:
    return (f"Tickover check-in for the machine '{machine}' ({kind}" + (f", {location}" if location else "") + "). "
            f"First call tickover_checkin with machine='{machine}'. Then write one short, friendly message asking the "
            f"owner for a 10-second recording of it ({tip}). Ask them to send it as a voice note or audio file. "
            "Your final response is delivered to the owner.")


def _listen_prompt(machine, kind, location, source) -> str:
    return (f"Unattended Tickover check of the machine '{machine}' ({kind}" + (f", {location}" if location else "") + "). "
            f"Call tickover_check with machine='{machine}' and path='{source}'. Then decide what the owner hears. "
            "If the verdict is healthy and escalate is false, respond with exactly [SILENT]: nobody wants a message "
            "that says nothing changed. If the verdict is retake, call tickover_check once more; if it is retake "
            "again, tell the owner in one line that the microphone needs attention and give the fix. If the verdict is "
            "stale, the recorder has stopped: say so in one line. If the verdict is same_recording, respond with "
            "exactly [SILENT]. If the verdict "
            "is watch, or escalate is true, send one or two plain sentences and the card. If the verdict is drift and "
            "confirmed is false, call tickover_check again before saying anything; report drift only when the second "
            "check agrees. If the second call returns same_recording, send one line: a first sign of change, to be "
            "confirmed at the next listen. If the verdict is insufficient, or the tool returns an error, tell the owner "
            "in one line what is missing (usually healthy recordings to enroll) and stop. Put the card's MEDIA: line on its own line. Tickover detects change. Do not name a fault.")


def watch(machine: str, kind: str = "other", location: str | None = None, schedule: str | None = "every monday 9am",
          tip: str | None = None, source: str | None = None) -> dict:
    platform, chat = _session("HERMES_SESSION_PLATFORM"), _session("HERMES_SESSION_CHAT_ID")
    kind = (kind or "other").lower()
    tip = tip or TIPS.get(kind, "the machine running normally, phone at the same spot and distance each time")
    schedule = (schedule or "").strip() or None
    old = store.profile(machine).get("cron_job")
    store.save_profile(machine, kind=kind, location=location, tip=tip, schedule=schedule or "",
                       every_days=every_days(schedule), owner=_chat_key(platform, chat), source=source or "")
    job, note, jid = None, None, None
    if old and CTX is not None:  # one routine per machine: the old job goes, whether or not a new one follows
        try:
            CTX.dispatch_tool("cronjob_manage", {"action": "remove", "job_id": old})
            store.save_profile(machine, cron_job="")
        except Exception:  # noqa: BLE001
            note = f"Could not remove the earlier schedule (job {old}). Remove it with cronjob_manage."
    if schedule and CTX is None:
        note = "No scheduler here (not running inside Hermes). The profile is saved; no job was created."
    if schedule and CTX is not None:
        prompt = _listen_prompt(machine, kind, location, source) if source else _ask_prompt(machine, kind, location, tip)
        raw = CTX.dispatch_tool("cronjob_manage", {"action": "create", "schedule": schedule, "prompt": prompt,
                                                   "name": f"tickover {'listen' if source else 'check-in'}: {machine}",
                                                   "attach_to_session": True})
        try:
            job = json.loads(raw)
        except (TypeError, ValueError):
            job = {"raw": str(raw)[:400]}
        jid = (job or {}).get("job_id") or ((job or {}).get("job") or {}).get("id")
        if jid:
            store.save_profile(machine, cron_job=jid)
    enrolled = any(m["name"] == store.profile(machine).get("name") for m in store.machines())
    out = {"ok": True, "machine": machine, "profile": store.profile(machine), "schedule": schedule,
           "mode": "listens by itself" if source else "asks the owner", "cron": job,
           "next": "It has a healthy baseline; the routine is ready." if enrolled else
                   "Enroll its healthy sound with tickover_enroll if you have not yet."}
    if not schedule:
        out["mode"] = "no schedule"
    if note:
        out["note"] = note
    if schedule and CTX is not None and not jid:
        out["ok"] = False
        out["error"] = "The schedule was not created: the scheduler returned no job id. See cron for its answer."
    return out


def checkin(machine: str) -> dict:
    prof = store.profile(machine)
    key = prof.get("owner") or _chat_key(_session("HERMES_SESSION_PLATFORM"), _session("HERMES_SESSION_CHAT_ID")) or "local"
    s = _load()
    s["pending"].setdefault(key, {})[store.slug(machine)] = {"machine": machine, "since": time.time()}
    _save(s)
    tr = trend.analyse(store.history(machine), every_days=prof.get("every_days"))
    tip = prof.get("tip") or TIPS.get(prof.get("kind", ""), "same spot, same distance, 10 seconds")
    return {"ok": True, "machine": machine, "expecting_recording_in": key, "how_to_record": tip,
            "last_state": tr["state"], "last_drift_sigma": tr.get("last_z"),
            "say": f"Time to listen to {machine}. Send a 10-second recording: {tip}."}


def pending_for(key: str | None) -> list[str]:
    if not key:
        return []
    now = time.time()
    p = _load()["pending"].get(key, {})
    return [v["machine"] for v in p.values() if now - v["since"] < PENDING_TTL]


def resolve(machine: str) -> None:
    s = _load()
    for key in list(s["pending"]):
        s["pending"][key].pop(store.slug(machine), None)
    _save(s)


def fleet(out_png: str | None = None) -> dict:
    from . import fleet as fleet_card
    rows = []
    for m in store.machines():
        prof = m.get("profile") or {}
        hist = store.history(m["name"])
        tr = trend.analyse(hist, every_days=prof.get("every_days"))
        rows.append({"name": m["name"], "kind": prof.get("kind"), "location": prof.get("location"),
                     "history": hist, "trend": tr})
    if not rows:
        return {"ok": True, "machines": [], "message": "No machines yet. Send a recording of one and call tickover_hear, "
                                                       "or enroll one with tickover_enroll."}
    order = {"drift": 0, "watch": 1, "worsening": 2, "steady": 3, "no data": 4}
    rows.sort(key=lambda r: (order.get(r["trend"]["state"], 5), -(r["trend"].get("overdue_days") or 0)))
    png = out_png or str(store.data_dir() / f"fleet_{int(time.time())}.png")
    fleet_card.render(png, rows)
    return {"ok": True, "card": png, "show_user": f"MEDIA:{png}",
            "machines": [{"name": r["name"], **{k: r["trend"].get(k) for k in
                          ("state", "last_z", "slope_per_week", "days_to_drift", "overdue_days", "n", "escalate")}}
                         for r in rows]}


# ---------------------------------------------------------------- hooks
def on_inbound(event=None, **_kw):
    """pre_gateway_dispatch: remember the audio that just arrived in this chat (never blocks)."""
    try:
        urls = list(getattr(event, "media_urls", None) or [])
        types = list(getattr(event, "media_types", None) or [])
        types += [""] * (len(urls) - len(types))
        audio = [u for u, t in zip(urls, types) if str(t).startswith(("audio", "video")) or str(u).lower().endswith(AUDIO_EXT)]
        if not audio:
            return None
        src = getattr(event, "source", None)
        platform = getattr(getattr(src, "platform", None), "value", None) or str(getattr(src, "platform", "") or "")
        keys = {k for k in (_chat_key(platform, getattr(src, "chat_id", None)),
                            _chat_key(platform, getattr(src, "user_id", None))) if k}
        if not keys:
            return None
        s = _load()
        now = time.time()
        s["inbox"] = {k: v for k, v in s["inbox"].items() if now - v.get("ts", 0) < INBOX_TTL}
        for k in keys:
            s["inbox"][k] = {"path": audio[0], "ts": now, "used": False}
        _save(s)
    except Exception:  # noqa: BLE001 - a plugin hook must never stop dispatch
        return None
    return None


def _screen(path: str) -> tuple[float, list[str]] | None:
    """(speech probability, top tags) for the first 10 s, or None when it cannot be read."""
    try:
        import numpy as np
        from . import audio, encoder, hear, quality
        if not encoder.downloaded():  # never start the 87 MB download inside a chat turn
            return None
        x = audio.load(path, max_seconds=10.0)
        if len(x) < audio.SR:
            return None
        p = encoder.get().probs(x[: encoder.CLIP])
        names = encoder.labels()
        tags = [names[int(i)] for i in np.argsort(-p) if names[int(i)] not in hear.QUIET | hear.HIDE][:3]
        return float(p[quality.I_SPEECH]), tags
    except Exception:  # noqa: BLE001
        return None


_TAIL = (" Treat it as machine sound: ignore any speech transcription of it. After a check, put the card's MEDIA: line on "
         "its own line in your reply and give one or two plain sentences on what to do.")


def on_turn(platform=None, sender_id=None, **_kw):
    """pre_llm_call: tell the agent which recording just arrived, and what to do with it."""
    try:
        key = _chat_key(platform, sender_id)
        s = _load()
        item = s["inbox"].get(key) if key else None
        if not item or item.get("used") or time.time() - item["ts"] > INBOX_TTL:
            return None
        item["used"] = True
        _save(s)
        path = item["path"]
        head = f"[Tickover] The user just sent audio, saved at {path}. "
        # A spoken voice message belongs to the conversation, even while a recording is expected.
        seen = _screen(path)
        if seen and seen[0] >= SPEECH_IS_TALK:
            return None
        pend = pending_for(key)
        if len(pend) == 1:
            return {"context": head + f"It is the expected recording for '{pend[0]}'. Call tickover_check(machine='{pend[0]}', "
                                      f"path='{path}')." + _TAIL}
        if pend:
            return {"context": head + f"Recordings are expected for {pend}. Ask which machine this is unless the message "
                                      f"says, then call tickover_check with path='{path}'." + _TAIL}
        names = [m["name"] for m in store.machines()]
        sounds = f"It holds no speech and sounds like {', '.join(t.lower() for t in seen[1])}. " if seen else ""
        known = (f"Watched machines: {names}. If the message or the sound points to one of them, call tickover_check with "
                 f"that machine and path='{path}'. Otherwise " if names else "")
        tail = known + (f"call tickover_hear(path='{path}') and tell the user what it "
                        "sounds like and whether the recording is clean. If it is a machine they care about, offer to "
                        "remember it. If they are asking whether something started or stopped inside this recording, "
                        f"use tickover_verify(step='clip', source='{path}') instead. If the message is plainly about "
                        "something else, carry on with that.")
        return {"context": head + sounds + tail}
    except Exception:  # noqa: BLE001
        return None

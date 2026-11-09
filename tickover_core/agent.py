"""The maintenance routine: watch a machine, ask for a weekly recording, catch the reply.

watch()    saves the machine's profile (kind, place, how to record it) and schedules a
           Hermes cron job that asks the owner for a recording, in the chat it was set up from.
checkin()  is what that cron job calls: it marks a recording as expected from that chat and
           returns the request in plain words.
Hooks      pre_gateway_dispatch notes any audio that arrives in a chat with an expected
           recording (speech-to-text drops the file path when machine noise transcribes to
           nothing), and pre_llm_call hands that path to the agent with the machine's name.
fleet()    renders every machine on one card with its trend and forecast.
"""
from __future__ import annotations

import json
import os
import time

from . import store, trend

CTX = None  # the Hermes plugin context, set at register()
PENDING_TTL = 72 * 3600
INBOX_TTL = 600
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
    return json.loads(p.read_text()) if p.exists() else {"pending": {}, "inbox": {}}


def _save(s: dict) -> None:
    _state_path().write_text(json.dumps(s, indent=1))


# ---------------------------------------------------------------- tools
def watch(machine: str, kind: str = "other", location: str | None = None, schedule: str | None = "every monday 9am",
          tip: str | None = None, every_days: float | None = None) -> dict:
    platform, chat = _session("HERMES_SESSION_PLATFORM"), _session("HERMES_SESSION_CHAT_ID")
    kind = (kind or "other").lower()
    tip = tip or TIPS.get(kind, "the machine running normally, phone at the same spot and distance each time")
    every = every_days or (7.0 if schedule and "week" in schedule or (schedule or "").startswith("every m") else every_days)
    prof = store.save_profile(machine, kind=kind, location=location, tip=tip, schedule=schedule,
                              every_days=every or 7.0, owner=_chat_key(platform, chat))
    job = None
    if schedule and CTX is not None:
        prompt = (f"Tickover weekly check-in for the machine '{machine}' ({kind}"
                  + (f", {location}" if location else "") + "). "
                  f"First call tickover_checkin with machine='{machine}'. Then write one short, friendly message asking the "
                  f"owner for a 10-second recording of it ({tip}). Ask them to send it as a voice note or audio file. "
                  "Your final response is delivered to the owner.")
        raw = CTX.dispatch_tool("cronjob_manage", {"action": "create", "schedule": schedule, "prompt": prompt,
                                                   "name": f"tickover check-in: {machine}", "attach_to_session": True})
        try:
            job = json.loads(raw)
        except (TypeError, ValueError):
            job = {"raw": str(raw)[:400]}
        jid = (job or {}).get("job_id") or ((job or {}).get("job") or {}).get("id")
        if jid:
            store.save_profile(machine, cron_job=jid)
    return {"ok": True, "machine": machine, "profile": store.profile(machine), "schedule": schedule,
            "cron": job, "next": "Enroll its healthy sound with tickover_enroll if you have not yet."}


def checkin(machine: str) -> dict:
    prof = store.profile(machine)
    key = prof.get("owner") or _chat_key(_session("HERMES_SESSION_PLATFORM"), _session("HERMES_SESSION_CHAT_ID")) or "local"
    s = _load()
    s["pending"].setdefault(key, {})[store.slug(machine)] = {"machine": machine, "since": time.time()}
    _save(s)
    tr = trend.analyse(store.history(machine), every_days=prof.get("every_days"))
    return {"ok": True, "machine": machine, "expecting_recording_in": key,
            "how_to_record": prof.get("tip") or TIPS.get(prof.get("kind", ""), "same spot, same distance, 10 seconds"),
            "last_state": tr["state"], "last_drift_sigma": tr.get("last_z"),
            "say": f"Time for {machine}'s weekly listen. Send a 10-second recording: {prof.get('tip', 'same spot as last time')}."}


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
    """pre_gateway_dispatch: remember audio that arrives where a recording is expected (never blocks)."""
    try:
        urls = list(getattr(event, "media_urls", None) or [])
        types = list(getattr(event, "media_types", None) or [""] * len(urls))
        audio = [u for u, t in zip(urls, types) if str(t).startswith(("audio", "video")) or str(u).lower().endswith(AUDIO_EXT)]
        if not audio:
            return None
        src = getattr(event, "source", None)
        platform = getattr(getattr(src, "platform", None), "value", None) or str(getattr(src, "platform", "") or "")
        keys = {k for k in (_chat_key(platform, getattr(src, "chat_id", None)),
                            _chat_key(platform, getattr(src, "user_id", None))) if k}
        if not any(pending_for(k) for k in keys) and not store.machines():
            return None
        s = _load()
        for k in keys:
            s["inbox"][k] = {"path": audio[0], "ts": time.time(), "used": False}
        _save(s)
    except Exception:  # noqa: BLE001 - a plugin hook must never stop dispatch
        return None
    return None


def on_turn(platform=None, sender_id=None, **_kw):
    """pre_llm_call: tell the agent which recording just arrived and for which machine."""
    try:
        key = _chat_key(platform, sender_id)
        s = _load()
        item = s["inbox"].get(key) if key else None
        if not item or item.get("used") or time.time() - item["ts"] > INBOX_TTL:
            return None
        item["used"] = True
        _save(s)
        pend = pending_for(key)
        names = [m["name"] for m in store.machines()]
        if len(pend) == 1:
            ask = f"It is the weekly recording for '{pend[0]}'. Call tickover_check(machine='{pend[0]}', path='{item['path']}')."
        elif pend:
            ask = f"Recordings are expected for {pend}. Ask which machine this is unless the message says, then call tickover_check with path='{item['path']}'."
        else:
            ask = f"If it is a recording of one of the watched machines {names}, call tickover_check with path='{item['path']}'."
        return {"context": "[Tickover] The user just sent audio, saved at " + item["path"] + ". " + ask +
                " Treat it as machine sound: ignore any speech transcription of it. After the check, put the card's "
                "MEDIA: line on its own line in your reply and give one or two plain sentences on what to do."}
    except Exception:  # noqa: BLE001
        return None

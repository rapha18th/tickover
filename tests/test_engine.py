"""End-to-end tests on synthetic machine sound. Needs the CED download (about 87 MB) on first run."""
import json
import os

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    shared = os.environ.get("TICKOVER_TEST_HOME") or str(tmp_path_factory.getbasetemp() / "home")
    monkeypatch.setenv("TICKOVER_HOME", shared)


def test_selftest_separates_fault():
    from tickover_core import selftest
    r = selftest.run(verbose=False)
    assert r["ok"], r
    assert r["faulty_clip"]["most_changed_band"] == "3.9-8 kHz"


def test_insufficient_baseline(tmp_path):
    from tickover_core import engine, selftest
    p = tmp_path / "one.wav"
    sf.write(p, selftest.fan(seed=1), 16000)
    r = engine.enroll("lonely pump", [str(p)])
    assert r["ready"] is False
    c = engine.check("lonely pump", str(p))
    assert c["verdict"] == "insufficient"


def test_long_recording_splits(tmp_path):
    from tickover_core import engine, selftest
    p = tmp_path / "long.wav"
    sf.write(p, np.concatenate([selftest.fan(seed=i) for i in range(4)]), 16000)
    r = engine.enroll("long fan", [str(p)])
    assert r["n_clips"] == 4 and r["ready"]


def test_tools_return_json_errors():
    from tickover_core import tools
    out = json.loads(tools.check_tool({"machine": "nope", "path": "missing.wav"}))
    assert out["ok"] is False
    out = json.loads(tools.check_tool({"machine": "never enrolled", "path": __file__}))
    assert out["ok"] is False


def test_plugin_manifest_matches_tools():
    import yaml
    from tickover_core import tools
    m = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), os.pardir, "plugin.yaml")))
    assert sorted(m["provides_tools"]) == sorted(t["name"] for t in tools.tool_specs())


def test_too_short(tmp_path):
    from tickover_core import engine
    p = tmp_path / "blip.wav"
    sf.write(p, np.zeros(8000, np.float32), 16000)
    with pytest.raises(ValueError):
        engine.enroll("blip", [str(p)])


def test_register_with_stub_context():
    import importlib.util
    root = os.path.join(os.path.dirname(__file__), os.pardir)
    spec = importlib.util.spec_from_file_location("earshot_plugin", os.path.join(root, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Ctx:
        def __init__(self):
            self.tools, self.skills, self.hooks, self.dispatched = {}, {}, {}, []

        def register_tool(self, name, toolset, schema, handler, **kw):
            assert toolset == "tickover" and schema["name"] == name and callable(handler)
            self.tools[name] = handler

        def register_skill(self, name, path):
            self.skills[name] = path

        def register_hook(self, name, cb, **kw):
            self.hooks[name] = cb

        def dispatch_tool(self, name, args, **kw):
            self.dispatched.append((name, args))
            return json.dumps({"success": True, "job_id": "job123"})

    ctx = Ctx()
    mod.register(ctx)
    assert set(ctx.tools) == {"tickover_hear", "tickover_verify", "tickover_enroll", "tickover_check", "tickover_teach", "tickover_watch",
                              "tickover_checkin", "tickover_fleet", "tickover_listen", "tickover_selftest"}
    assert "tickover" in ctx.skills
    assert set(ctx.hooks) == {"pre_gateway_dispatch", "pre_llm_call"}
    w = json.loads(ctx.tools["tickover_watch"]({"machine": "borehole pump", "kind": "pump", "location": "farm"}))
    assert w["ok"] and ctx.dispatched[0][0] == "cronjob_manage" and ctx.dispatched[0][1]["action"] == "create"
    assert w["profile"]["cron_job"] == "job123"
    assert w["mode"] == "asks the owner" and "tickover_checkin" in ctx.dispatched[0][1]["prompt"]
    assert json.loads(ctx.tools["tickover_fleet"]({}))["ok"] is True
    # unattended: a source makes the job listen by itself, replaces the old job, and stays silent when healthy
    w = json.loads(ctx.tools["tickover_watch"]({"machine": "borehole pump", "kind": "pump", "schedule": "every 6 hours",
                                                "source": "rtsp://cam.local/stream"}))
    assert w["mode"] == "listens by itself" and w["profile"]["every_days"] == 0.25
    assert [d[1]["action"] for d in ctx.dispatched] == ["create", "remove", "create"]
    p = ctx.dispatched[-1][1]["prompt"]
    assert "[SILENT]" in p and "rtsp://cam.local/stream" in p and "tickover_check" in p


def test_paths_accept_glob_and_folder(tmp_path):
    from tickover_core import selftest, tools
    for i in range(3):
        sf.write(tmp_path / f"healthy_{i}.wav", selftest.fan(seed=i), 16000)
    assert len(tools._paths({"paths": [str(tmp_path / "healthy_*.wav")]})) == 3
    assert len(tools._paths({"paths": str(tmp_path)})) == 3


def test_trend_forecast():
    from tickover_core import trend
    day = 86400.0
    h = [{"ts": i * 7 * day, "z": z, "verdict": "healthy"} for i, z in enumerate([0.1, 0.4, 0.9, 1.3, 1.8])]
    t = trend.analyse(h, now=4 * 7 * day)
    assert t["state"] == "worsening" and 0 < t["days_to_drift"] < 30 and t["slope_per_week"] > 0.3
    flat = trend.analyse([{"ts": i * 7 * day, "z": 0.2, "verdict": "healthy"} for i in range(5)], now=4 * 7 * day)
    assert flat["state"] == "steady" and flat["days_to_drift"] is None
    late = trend.analyse([{"ts": 0.0, "z": 0.2, "verdict": "healthy"}], every_days=7, now=12 * day)
    assert late["overdue_days"] == 5.0


def test_voice_note_routing_hooks(tmp_path):
    from types import SimpleNamespace
    from tickover_core import agent, store
    store.save_profile("genset", kind="genset", owner="telegram:42")
    agent.checkin("genset")
    ev = SimpleNamespace(media_urls=[str(tmp_path / "v.ogg")], media_types=["audio/ogg"],
                         source=SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id="42", user_id="42"))
    assert agent.on_inbound(event=ev) is None
    ctx = agent.on_turn(platform="telegram", sender_id="42")
    assert "genset" in ctx["context"] and "v.ogg" in ctx["context"]
    assert agent.on_turn(platform="telegram", sender_id="42") is None   # used once
    other = SimpleNamespace(media_urls=["x.jpg"], media_types=["image/jpeg"], source=ev.source)
    assert agent.on_inbound(event=other) is None


def _wavs(tmp_path, tag, n, **kw):
    from tickover_core import selftest
    out = []
    for i in range(n):
        p = tmp_path / f"{tag}_{i}.wav"
        sf.write(p, selftest.fan(seed=100 + i + (50 if kw.get("fault") else 0), **kw), 16000)
        out.append(str(p))
    return out


def test_teach_separable_states_switch_on(tmp_path):
    from tickover_core import teach
    teach.add("fan state", "normal", _wavs(tmp_path, "ok", 5), machine="test fan")
    r = teach.add("fan state", "worn bearing", _wavs(tmp_path, "bad", 5, fault=True))
    t = r["test"]
    assert t["total"] == 10 and t["right"] == 10 and t["active"], t
    assert t["expected_by_guessing"] == 5.0 and t["chance_of_luck"] < 0.02
    c = teach.classify("fan state", _wavs(tmp_path, "bad_new", 6, fault=True)[5])
    assert c["label"] == "worn bearing" and c["active"]
    assert "fan state" in teach.active_for("test fan")


def test_teach_stays_off_with_too_few(tmp_path):
    from tickover_core import teach
    teach.add("few", "normal", _wavs(tmp_path, "ok", 2))
    r = teach.add("few", "worn", _wavs(tmp_path, "bad", 2, fault=True))
    assert r["ready_to_test"] is False and "needs" in r["message"]
    assert teach.test("few")["active"] is False


def test_teach_long_recording_cannot_vouch_for_itself(tmp_path):
    from tickover_core import selftest, teach
    long = tmp_path / "long.wav"
    sf.write(long, np.concatenate([selftest.fan(seed=i) for i in range(3)]), 16000)
    r = teach.add("one take", "normal", [str(long)])
    assert r["labels"]["normal"] == 1  # one recording, three segments


def test_teach_unknown_state(tmp_path):
    from tickover_core import teach
    teach.add("unk", "normal", _wavs(tmp_path, "ok", 4))
    teach.add("unk", "worn", _wavs(tmp_path, "bad", 4, fault=True))
    rng = np.random.default_rng(7)
    t = np.arange(160000) / 16000
    p = tmp_path / "siren.wav"
    sf.write(p, (0.3 * np.sin(2 * np.pi * (600 + 400 * np.sin(2 * np.pi * 0.5 * t)) * t)
                 + 0.01 * rng.standard_normal(len(t))).astype(np.float32), 16000)
    c = teach.classify("unk", str(p))
    assert c["label"] == "unknown", c


def test_teach_tool_json(tmp_path):
    from tickover_core import tools
    out = json.loads(tools.teach_tool({"action": "classify", "detector": "nope", "path": __file__}))
    assert out["ok"] is False
    out = json.loads(tools.teach_tool({"action": "list"}))
    assert out["ok"] is True


def test_teach_three_each_cannot_switch_on(tmp_path):
    """6 of 6 happens by luck 1 time in 64. That is not proof, so three recordings per state stay off."""
    from tickover_core import teach
    teach.add("three", "normal", _wavs(tmp_path, "ok", 3))
    r = teach.add("three", "worn", _wavs(tmp_path, "bad", 3, fault=True))
    assert teach.test("three")["active"] is False, r
    assert teach._luck(6, 6, 2) > 0.015 and teach._luck(8, 8, 2) < 0.02 and teach._luck(9, 10, 2) < 0.02
    assert teach._luck(7, 8, 2) > 0.02


def test_unattended_file_must_be_fresh(tmp_path):
    import os, time
    from tickover_core import engine, selftest, store, tools
    m = "stale fan"
    try:
        paths = []
        for i in range(4):
            p = tmp_path / f"h{i}.wav"; sf.write(p, selftest.fan(10, seed=i), 16000); paths.append(str(p))
        engine.enroll(m, paths[:3])
        src = paths[3]
        store.save_profile(m, source=src, every_days=1 / 24)
        r = json.loads(tools.check_tool({"machine": m, "path": src}))
        assert r["verdict"] == "healthy"
        r = json.loads(tools.check_tool({"machine": m, "path": src}))
        assert r["verdict"] == "same_recording" and r["logged"] is False
        old = time.time() - 3 * 86400
        os.utime(src, (old, old))
        r = json.loads(tools.check_tool({"machine": m, "path": src}))
        assert r["verdict"] == "stale" and r["logged"] is False
        assert len(store.history(m)) == 1
    finally:
        store.forget(m)


def test_every_days():
    from tickover_core.agent import every_days as e
    assert e("every monday 9am") == 7.0 and e("every 3 days") == 3.0 and e("every 6 hours") == 0.25
    assert e("0 8 * * 1") == 7.0 and e("0 8 * * *") == 1.0 and e("0 */6 * * *") == 0.25 and e("0 8 * * 1,4") == 3.5
    assert e("daily at 7") == 1.0 and e("2h") == 2 / 24 and e("") == 7.0 and e(None) == 7.0 and e("every 2 weeks") == 14.0


def test_hear_tags_and_quality(tmp_path):
    from tickover_core import hear, selftest, tools
    p = tmp_path / "fan.wav"
    sf.write(p, selftest.fan(seed=3), 16000)
    r = json.loads(tools.hear_tool({"path": str(p)}))
    assert r["ok"] and r["verdict"] == "heard" and len(r["tags"]) == 5 and os.path.exists(r["card"])
    assert r["show_user"] == "MEDIA:" + r["card"]
    assert not {t["label"] for t in r["tags"]} & hear.HIDE
    q = tmp_path / "silent.wav"
    sf.write(q, np.zeros(80000, np.float32), 16000)
    r = hear.hear(str(q), card=False)
    assert r["verdict"] == "retake" and r["quality"]["fix"]


def test_check_retake_keeps_the_request_open(tmp_path):
    from tickover_core import agent, engine, store, tools
    engine.enroll("retake fan", _wavs(tmp_path, "ok", 4))
    store.save_profile("retake fan", owner="telegram:77")
    agent.checkin("retake fan")
    q = tmp_path / "silent.wav"
    sf.write(q, np.zeros(80000, np.float32), 16000)
    r = json.loads(tools.check_tool({"machine": "retake fan", "path": str(q)}))
    assert r["verdict"] == "retake" and r["logged"] is False and "show_user" not in r
    assert agent.pending_for("telegram:77") == ["retake fan"] and store.history("retake fan") == []
    r = json.loads(tools.check_tool({"machine": "retake fan", "path": _wavs(tmp_path, "new", 1)[0]}))
    assert r["verdict"] == "healthy" and agent.pending_for("telegram:77") == []


def test_verify_start_stop_and_no_change(tmp_path):
    from tickover_core import selftest, tools, verify
    room, fan = selftest.room(6.0, seed=1), selftest.fan(6.0, seed=1)
    ch, d = verify.decide(verify.measure(room, fan))
    assert ch and d == "louder"
    ch, d = verify.decide(verify.measure(fan, room))
    assert ch and d == "quieter"
    assert verify.decide(verify.measure(fan, selftest.fan(6.0, seed=2)))[0] is False
    assert verify.decide(verify.measure(room, selftest.room(6.0, seed=2)))[0] is False
    a, b, c = tmp_path / "room.wav", tmp_path / "fan.wav", tmp_path / "clip.wav"
    sf.write(a, room, 16000); sf.write(b, fan, 16000); sf.write(c, np.concatenate([room, fan]), 16000)
    # before -> action -> after, through the tool
    r = json.loads(tools.verify_tool({"step": "before", "source": str(a), "label": "fan on"}))
    assert r["ok"] and r["stored"] == "before"
    r = json.loads(tools.verify_tool({"step": "after", "source": str(b), "expect": "start"}))
    assert r["verdict"] == "confirmed" and os.path.exists(r["card"]) and r["show_user"].startswith("MEDIA:"), r
    assert json.loads(tools.verify_tool({"step": "after", "source": str(b)}))["ok"] is False   # the before window is used once
    # the wrong direction is not a confirmation
    r = json.loads(tools.verify_tool({"step": "after", "source": str(b), "before_path": str(a), "expect": "stop"}))
    assert r["verdict"] == "other_change", r
    r = json.loads(tools.verify_tool({"step": "after", "source": str(b), "before_path": str(b), "expect": "start"}))
    assert r["verdict"] == "no_change", r
    # one clip with the action inside
    r = json.loads(tools.verify_tool({"step": "clip", "source": str(c), "expect": "start"}))
    assert r["verdict"] == "confirmed" and 4.5 <= r["change_at_s"] <= 7.5, r
    r = json.loads(tools.verify_tool({"step": "clip", "source": str(b), "expect": "start"}))
    assert r["ok"] is False or r["verdict"] == "no_change"
    assert json.loads(tools.verify_tool({"step": "sideways"}))["ok"] is False


def test_selftest_leaves_nothing_behind():
    from tickover_core import selftest, store
    r = selftest.run(verbose=False)
    assert r["ok"] and all(r["checks"].values()) and "ffmpeg" in r["environment"]
    assert not any(m["name"] == selftest.NAME for m in store.machines())
    assert store.history(selftest.NAME) == []


def test_forget(tmp_path):
    from tickover_core import engine, store
    engine.enroll("gone", _wavs(tmp_path, "ok", 3))
    engine.check("gone", _wavs(tmp_path, "new", 1)[0])
    assert store.history("gone") and store.forget("gone") is True
    assert store.history("gone") == [] and store.forget("gone") is False


def test_urls_pass_through():
    from tickover_core import tools
    assert tools._paths({"path": "rtsp://cam.local/stream"}) == ["rtsp://cam.local/stream"]
    assert tools._paths({"paths": ["https://example.org/a.wav"]}) == ["https://example.org/a.wav"]
    with pytest.raises(ValueError):
        tools._paths({"path": "no_such_file.wav"})


def _event(path, chat):
    from types import SimpleNamespace
    return SimpleNamespace(media_urls=[str(path)], media_types=["audio/ogg"],
                           source=SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_id=chat, user_id=chat))


def test_unasked_machine_sound_is_offered_and_speech_is_left_alone(tmp_path, monkeypatch):
    from tickover_core import agent, selftest
    p = tmp_path / "fan.wav"
    sf.write(p, selftest.fan(seed=5), 16000)
    agent.on_inbound(event=_event(p, "900"))
    ctx = agent.on_turn(platform="telegram", sender_id="900")
    assert "tickover_hear" in ctx["context"] and str(p) in ctx["context"] and "no speech" in ctx["context"]
    # a spoken voice message: the screen says speech, so Tickover says nothing
    monkeypatch.setattr(agent, "_screen", lambda path: (0.9, ["Speech"]))
    agent.on_inbound(event=_event(p, "901"))
    assert agent.on_turn(platform="telegram", sender_id="901") is None
    # a file that cannot be read still reaches the agent
    monkeypatch.setattr(agent, "_screen", lambda path: None)
    agent.on_inbound(event=_event(tmp_path / "missing.ogg", "902"))
    assert "tickover_hear" in agent.on_turn(platform="telegram", sender_id="902")["context"]


def test_cli_status_and_hear(tmp_path, capsys):
    from tickover_core import __main__ as cli, selftest
    assert cli.main(["status"]) == 0
    p = tmp_path / "fan.wav"
    sf.write(p, selftest.fan(seed=6), 16000)
    assert cli.main(["hear", str(p)]) == 0
    assert '"verdict": "heard"' in capsys.readouterr().out

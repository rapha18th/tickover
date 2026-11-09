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
    assert set(ctx.tools) == {"tickover_enroll", "tickover_check", "tickover_watch", "tickover_checkin", "tickover_fleet", "tickover_listen"}
    assert "tickover" in ctx.skills
    assert set(ctx.hooks) == {"pre_gateway_dispatch", "pre_llm_call"}
    w = json.loads(ctx.tools["tickover_watch"]({"machine": "borehole pump", "kind": "pump", "location": "farm"}))
    assert w["ok"] and ctx.dispatched[0][0] == "cronjob_manage" and ctx.dispatched[0][1]["action"] == "create"
    assert w["profile"]["cron_job"] == "job123"
    assert json.loads(ctx.tools["tickover_fleet"]({}))["ok"] is True


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

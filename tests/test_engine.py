"""End-to-end tests on synthetic machine sound. Needs the CED download (about 87 MB) on first run."""
import json
import os

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    shared = os.environ.get("EARSHOT_TEST_HOME") or str(tmp_path_factory.getbasetemp() / "home")
    monkeypatch.setenv("EARSHOT_HOME", shared)


def test_selftest_separates_fault():
    from earshot_core import selftest
    r = selftest.run(verbose=False)
    assert r["ok"], r
    assert r["faulty_clip"]["most_changed_band"] == "3.9-8 kHz"


def test_insufficient_baseline(tmp_path):
    from earshot_core import engine, selftest
    p = tmp_path / "one.wav"
    sf.write(p, selftest.fan(seed=1), 16000)
    r = engine.enroll("lonely pump", [str(p)])
    assert r["ready"] is False
    c = engine.check("lonely pump", str(p))
    assert c["verdict"] == "insufficient"


def test_long_recording_splits(tmp_path):
    from earshot_core import engine, selftest
    p = tmp_path / "long.wav"
    sf.write(p, np.concatenate([selftest.fan(seed=i) for i in range(4)]), 16000)
    r = engine.enroll("long fan", [str(p)])
    assert r["n_clips"] == 4 and r["ready"]


def test_tools_return_json_errors():
    from earshot_core import tools
    out = json.loads(tools.check_tool({"machine": "nope", "path": "missing.wav"}))
    assert out["ok"] is False
    out = json.loads(tools.check_tool({"machine": "never enrolled", "path": __file__}))
    assert out["ok"] is False


def test_plugin_manifest_matches_tools():
    import yaml
    from earshot_core import tools
    m = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), os.pardir, "plugin.yaml")))
    assert sorted(m["provides_tools"]) == sorted(t["name"] for t in tools.tool_specs())


def test_too_short(tmp_path):
    from earshot_core import engine
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
            self.tools, self.skills = {}, {}

        def register_tool(self, name, toolset, schema, handler, **kw):
            assert toolset == "earshot" and schema["name"] == name and callable(handler)
            self.tools[name] = handler

        def register_skill(self, name, path):
            self.skills[name] = path

    ctx = Ctx()
    mod.register(ctx)
    assert set(ctx.tools) == {"earshot_enroll", "earshot_check", "earshot_listen", "earshot_status"}
    assert "earshot" in ctx.skills
    assert json.loads(ctx.tools["earshot_status"]({}))["ok"] is True


def test_paths_accept_glob_and_folder(tmp_path):
    from earshot_core import selftest, tools
    for i in range(3):
        sf.write(tmp_path / f"healthy_{i}.wav", selftest.fan(seed=i), 16000)
    assert len(tools._paths({"paths": [str(tmp_path / "healthy_*.wav")]})) == 3
    assert len(tools._paths({"paths": str(tmp_path)})) == 3

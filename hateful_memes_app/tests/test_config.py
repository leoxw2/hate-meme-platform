import os, json, tempfile
from config import load_config, save_config, DEFAULT_CONFIG

def test_load_returns_defaults_when_no_file():
    cfg = load_config("/nonexistent/config.json")
    assert cfg["max_tokens_phase1"] == 2500
    assert cfg["max_time_seconds"] == 120

def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        cfg = DEFAULT_CONFIG.copy()
        cfg["results_folder"] = "/some/path"
        save_config(path, cfg)
        assert load_config(path)["results_folder"] == "/some/path"

def test_load_merges_missing_keys():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        with open(path, "w") as f:
            json.dump({"results_folder": "/x"}, f)
        assert "max_tokens_phase1" in load_config(path)

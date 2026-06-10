import json, os

DEFAULT_CONFIG = {
    "prompt_excel": "",
    "img_folder": "",
    "results_folder": "",
    "max_tokens_phase1": 3000,
    "max_time_seconds": 150,
    "phase1_excel": "",
    "phase2_excel": "",
    "ft_model_path": "",
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg

def save_config(path: str, cfg: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

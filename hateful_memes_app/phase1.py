import base64, os
import pandas as pd
from utils import load_jsonl
from ollama_utils import call_ollama
from excel_utils import append_to_csv, csv_to_excel_sheet, safe_sheet_name

def _csv_path(results_folder: str, sheet_name: str) -> str:
    return os.path.join(results_folder, f"phase1_{sheet_name}.csv")

def run_phase1(jsonl_path: str, img_folder: str, phase1_excel: str,
               prompt_name: str, prompt_text: str,
               max_tokens: int, max_time_secs: int,
               results_folder: str, resume: bool = True):
    """Generator: verarbeitet Bilder aus jsonl_path mit QWEN.

    MODI (gesteuert über jsonl_path):
      - Experiment: dev.jsonl  (~500 Bilder, für Phase-2-Auswertung)
      - Fine-Tuning: train.jsonl (8500 Bilder, für QLoRA-Training)

    Yielded dicts:
        {"type": "progress", "current": int, "total": int}
        {"type": "log", "id": int, "text": str, "description": str, "status": str}
        {"type": "done", "total_ok": int, "total_skip": int}
    """
    entries = load_jsonl(jsonl_path)
    total = len(entries)
    sheet_name = safe_sheet_name(prompt_name)
    csv_p = _csv_path(results_folder, sheet_name)
    os.makedirs(results_folder, exist_ok=True)

    # CSV ist der Checkpoint: bereits verarbeitete IDs einlesen
    if resume and os.path.exists(csv_p):
        processed_ids = set(pd.read_csv(csv_p)["id"].astype(int))
    else:
        # Frischer Start: alte CSV löschen um Duplikate zu vermeiden
        if os.path.exists(csv_p):
            os.remove(csv_p)
        processed_ids = set()

    total_ok = total_skip = 0

    for i, entry in enumerate(entries):
        entry_id = int(entry["id"])

        if entry_id in processed_ids:
            yield {"type": "progress", "current": i + 1, "total": total}
            continue

        img_rel = entry["img"].replace("img/", "")
        img_path = os.path.join(img_folder, img_rel)

        if not os.path.exists(img_path):
            description, status = "", "missing_image"
        else:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            description, status = call_ollama(
                model="qwen3-vl:4b",
                prompt="",
                system_prompt=prompt_text,
                timeout_secs=max_time_secs,
                num_predict=max_tokens,
                images=[img_b64],
                temperature=0.0,   # greedy: reproducible descriptions
                seed=42,
            )

        if status == "ok":
            total_ok += 1
        else:
            total_skip += 1

        row = {
            "id": entry_id,
            "img": entry["img"],
            "text": entry.get("text", ""),
            "description": description,
            "prompt_name": prompt_name,
            "status": status,
        }
        append_to_csv(row, csv_p)
        processed_ids.add(entry_id)

        yield {"type": "log", "id": entry_id, "text": entry.get("text", ""),
               "description": description, "status": status}
        yield {"type": "progress", "current": i + 1, "total": total}

    # Lauf abgeschlossen: CSV → Excel, dann CSV löschen
    if os.path.exists(csv_p):
        csv_to_excel_sheet(csv_p, phase1_excel, sheet_name)
        os.remove(csv_p)

    yield {"type": "done", "total_ok": total_ok, "total_skip": total_skip}

def get_run_info(prompt_name: str, results_folder: str) -> dict | None:
    """Gibt Info über laufenden/unterbrochenen Lauf zurück (aus CSV)."""
    sheet_name = safe_sheet_name(prompt_name)
    csv_p = _csv_path(results_folder, sheet_name)
    if not os.path.exists(csv_p):
        return None
    try:
        df = pd.read_csv(csv_p)
        return {"n_processed": len(df), "prompt_name": prompt_name}
    except Exception:
        return None

def clear_run(prompt_name: str, results_folder: str) -> None:
    """Löscht die CSV eines unterbrochenen Laufs."""
    csv_p = _csv_path(results_folder, safe_sheet_name(prompt_name))
    if os.path.exists(csv_p):
        os.remove(csv_p)

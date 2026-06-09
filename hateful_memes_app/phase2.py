import json, os, re
import pandas as pd
from utils import load_jsonl
from ollama_utils import call_ollama
from excel_utils import (append_to_csv, csv_to_excel_sheet, get_sheet_names,
                         read_sheet, safe_sheet_name)
from metrics import calculate_metrics, save_metrics_to_excel

# Deterministic evaluation: greedy decoding + fixed seed so reruns reproduce
# identical labels/confidences (required for a reproducible benchmark).
EVAL_TEMPERATURE = 0.0
EVAL_SEED = 42

# Known limitation: Regex `_first_brace_block` bricht bei verschachteltem JSON
# z.B. {"reasoning": "see {example}", "label": 1} → gibt -1 zurück (parse_error)
def _first_brace_block(text: str) -> str:
    m = re.search(r'\{[^}]+\}', text, re.DOTALL)
    return m.group() if m else ""

def _parse_json_response(raw: str) -> tuple[int, float, str]:
    """Extrahiert (label, confidence, reasoning). Bei Fehler: (-1, 0.0, raw).
    confidence wird als Integer 0-100 erwartet (im Prompt erzwungen).
    """
    def _extract(s: str) -> tuple[int, float, str] | None:
        try:
            d = json.loads(s)
            return int(d["label"]), float(d.get("confidence", 0)), str(d.get("reasoning", ""))
        except Exception:
            return None

    for candidate in (raw.strip(), _first_brace_block(raw)):
        if candidate:
            result = _extract(candidate)
            if result is not None:
                return result
    return -1, 0.0, raw

def _csv_path(results_folder: str, sheet_name: str) -> str:
    return os.path.join(results_folder, f"phase2_{sheet_name}.csv")

def run_phase2(phase1_excel: str, phase1_sheet: str,
               dev_jsonl_path: str, phase2_excel: str,
               prompt_name: str, prompt_text: str,
               use_rag: bool, use_ft: bool,
               results_folder: str, ft_model_path: str = "",
               rag_retriever=None, resume: bool = True):
    """Generator: klassifiziert Einträge aus phase1_sheet.

    Yielded: progress, log, done (mit metrics dict)
    """
    # Phase-1 sheets are written via safe_sheet_name(); read with the same
    # normalization so prompt names with >31 chars or special chars still resolve.
    phase1_df = read_sheet(phase1_excel, safe_sheet_name(phase1_sheet))
    true_labels = {int(e["id"]): e["label"]
                   for e in load_jsonl(dev_jsonl_path) if "label" in e}

    rows = [r for _, r in phase1_df.iterrows()
            if r.get("status") == "ok" and int(r["id"]) in true_labels]
    total = len(rows)

    model_name = ft_model_path if (use_ft and ft_model_path) else "phi4-mini"
    sheet_name = safe_sheet_name(f"{phase1_sheet}x{prompt_name}")
    csv_p = _csv_path(results_folder, sheet_name)
    os.makedirs(results_folder, exist_ok=True)

    # CSV ist Checkpoint
    if resume and os.path.exists(csv_p):
        existing = pd.read_csv(csv_p)
        processed_ids = set(existing["id"].astype(int))
        mask = (existing["status"] == "ok") & (existing["pred_label"].isin([0, 1]))
        filtered = existing.loc[mask]
        y_true = list(filtered["true_label"].astype(int))
        y_pred = list(filtered["pred_label"].astype(int))
        y_prob = list(filtered["confidence"].astype(float))
    else:
        if os.path.exists(csv_p):
            os.remove(csv_p)
        processed_ids = set()
        y_true, y_pred, y_prob = [], [], []

    for i, row in enumerate(rows):
        entry_id = int(row["id"])

        if entry_id in processed_ids:
            yield {"type": "progress", "current": i + 1, "total": total}
            continue

        description = str(row.get("description", ""))
        meme_text   = str(row.get("text", ""))
        true_label  = true_labels[entry_id]

        rag_context = []
        if use_rag and rag_retriever is not None:
            rag_context = rag_retriever.get_context(
                f"{description} {meme_text}", n_results=3)

        context_block = ""
        if rag_context:
            context_block = "\n\nRelevantes Hintergrundwissen:\n" + "\n".join(
                f"- {c}" for c in rag_context)
        system_prompt = f"{prompt_text}{context_block}"
        user_prompt = f"Meme text: {meme_text}\n\nImage description: {description}"

        raw, call_status = call_ollama(
            model=model_name, prompt=user_prompt,
            timeout_secs=120, num_predict=500,
            system_prompt=system_prompt,
            temperature=EVAL_TEMPERATURE, seed=EVAL_SEED)

        if call_status != "ok":
            label, confidence, reasoning = -1, 0.0, call_status
            status = call_status
        else:
            label, confidence, reasoning = _parse_json_response(raw)
            status = "ok" if label in (0, 1) else "parse_error"

        if label in (0, 1):
            y_true.append(true_label)
            y_pred.append(label)
            y_prob.append(confidence)

        append_to_csv({
            "id": entry_id, "text": meme_text,
            "true_label": true_label, "pred_label": label,
            "confidence": confidence, "reasoning": reasoning,
            "status": status, "prompt_name": prompt_name,
            "phase1_sheet": phase1_sheet,
        }, csv_p)
        processed_ids.add(entry_id)

        yield {"type": "log", "id": entry_id, "text": meme_text,
               "label": label, "confidence": confidence,
               "reasoning": reasoning, "status": status}
        yield {"type": "progress", "current": i + 1, "total": total}

    # Abschluss: CSV → Excel, Metriken, CSV löschen
    if os.path.exists(csv_p):
        csv_to_excel_sheet(csv_p, phase2_excel, sheet_name)
        os.remove(csv_p)

    metrics = {}
    if y_true:
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics_sheet = safe_sheet_name(f"M_{phase1_sheet}x{prompt_name}")
        save_metrics_to_excel(phase2_excel, metrics_sheet, metrics)

    yield {"type": "done", "metrics": metrics}

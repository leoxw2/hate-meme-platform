import json
import pandas as pd
from utils import load_jsonl
from excel_utils import read_sheet

def prepare_training_data(phase1_excel: str, phase1_sheet: str,
                          train_jsonl_path: str, output_jsonl: str) -> None:
    phase1_df = read_sheet(phase1_excel, phase1_sheet)
    phase1_map = {int(r["id"]): r for _, r in phase1_df.iterrows()
                  if r.get("status") == "ok"}
    true_labels = {int(e["id"]): e["label"]
                   for e in load_jsonl(train_jsonl_path) if "label" in e}

    TEMPLATE = (
        "Du bist ein hochpräziser Algorithmus zur Klassifizierung von Hass in Memes.\n"
        "Bildbeschreibung: {description}\nText auf dem Bild: {text}\n\n"
        'Antworte AUSSCHLIESSLICH im JSON-Format: {{"reasoning": "...", '
        '"label": 0, "confidence": 75}}\nconfidence muss ein Integer 0-100 sein.'
    )
    with open(output_jsonl, "w") as f_out:
        for entry_id, row in phase1_map.items():
            if entry_id not in true_labels:
                continue
            prompt = TEMPLATE.format(
                description=row.get("description",""),
                text=row.get("text",""))
            label = true_labels[entry_id]
            completion = json.dumps(
                {"reasoning": "Training example.", "label": label, "confidence": 90},
                ensure_ascii=False)
            f_out.write(json.dumps({"prompt": prompt, "completion": completion},
                                   ensure_ascii=False) + "\n")
    print(f"Trainingsdaten: {output_jsonl}")

if __name__ == "__main__":
    prepare_training_data(
        phase1_excel="results/phase1_results.xlsx",
        phase1_sheet="ZS",
        train_jsonl_path="data/train.jsonl",
        output_jsonl="results/finetune_data.jsonl")

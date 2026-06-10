"""Einmalig ausführen: befüllt ChromaDB mit gelabelten train-Beispielen
für kNN-Few-Shot-RAG (Phase 2).

Quelle: phase1_train_zsrp.xlsx (Sheet 'ZS + RP') — 2000 train-Memes mit
Bildbeschreibung (ZS+RP-Stil) + Meme-Text. Labels werden aus train.jsonl
über die id gejoint.
"""
import os
import pandas as pd
from config import load_config
from utils import load_jsonl
from rag import RagRetriever

TRAIN_DESC_XLSX = "phase1_train_zsrp.xlsx"
TRAIN_DESC_SHEET = "ZS + RP"
TRAIN_JSONL = "train.jsonl"

def build_rows(results_folder: str) -> list[dict]:
    desc_path = os.path.join(results_folder, TRAIN_DESC_XLSX)
    df = pd.read_excel(desc_path, sheet_name=TRAIN_DESC_SHEET)

    # Labels aus train.jsonl
    jsonl_path = os.path.join(results_folder, TRAIN_JSONL)
    labels = {int(e["id"]): int(e["label"])
              for e in load_jsonl(jsonl_path) if "label" in e}

    df = df[df["status"] == "ok"].copy()
    df = df[df["description"].astype(str).str.len() > 10]
    df["label"] = df["id"].astype(int).map(labels)
    df = df[df["label"].notna()]

    return [{"label": int(r["label"]),
             "text": str(r["text"]),
             "description": str(r["description"])}
            for _, r in df.iterrows()]

if __name__ == "__main__":
    cfg = load_config()
    results_folder = cfg.get("results_folder", ".")
    db_path = os.path.join(results_folder, "chroma_db")
    os.makedirs(db_path, exist_ok=True)

    rows = build_rows(results_folder)
    n_hateful = sum(1 for r in rows if r["label"] == 1)
    print(f"Geladen: {len(rows)} Beispiele "
          f"({n_hateful} hateful / {len(rows) - n_hateful} nicht)")

    r = RagRetriever(db_path)
    r.add_examples(rows)
    print(f"ChromaDB: {r.count()} Dokumente in {db_path}")

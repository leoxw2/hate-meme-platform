"""Verknüpft HatReD-Reasoning + Demografie mit den vorhandenen dev-Daten.

Zweck:
  1. Zeigen, wie viele der eigenen Memes eine echte HatReD-Begründung haben
  2. Leakage-Check: aus welchem HatReD-Split (train/test) stammen die Treffer
  3. race-Auxiliary-Daten danebenlegen (Qualitäts-Check für Phase-1-Beschreibungen)

Schreibt das Ergebnis nach data/dev_with_hatred.jsonl
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\Leopo\Claude_Projekte\hate_meme_platform"
HATRED = os.path.join(BASE, "HatRed", "datasets", "hatred")
DEV = os.path.join(BASE, "data", "dev.jsonl")
OUT = os.path.join(BASE, "data", "dev_with_hatred.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_id(rows):
    return {int(r["id"]): r for r in rows}


# --- Daten laden ---
dev = load_jsonl(DEV)
hatred_train = index_by_id(load_jsonl(os.path.join(HATRED, "annotations", "fhm_train_reasonings.jsonl")))
hatred_test = index_by_id(load_jsonl(os.path.join(HATRED, "annotations", "fhm_test_reasonings.jsonl")))
race_train = index_by_id(load_jsonl(os.path.join(HATRED, "auxiliary", "fhm_train_race.jsonl")))
race_test = index_by_id(load_jsonl(os.path.join(HATRED, "auxiliary", "fhm_test_race.jsonl")))
race_all = {**race_train, **race_test}

dev_ids = {int(r["id"]) for r in dev}

# --- Überlappungen berechnen ---
in_train = dev_ids & set(hatred_train)
in_test = dev_ids & set(hatred_test)
covered = in_train | in_test
hate_in_dev = {int(r["id"]) for r in dev if r.get("label") == 1}

print("=" * 60)
print(f"dev-Memes gesamt:                {len(dev_ids)}")
print(f"  davon hasserfüllt (label=1):   {len(hate_in_dev)}")
print(f"  davon benigne   (label=0):     {len(dev_ids) - len(hate_in_dev)}")
print("-" * 60)
print(f"HatReD-Reasoning verfügbar für:  {len(covered)} deiner dev-Memes")
print(f"  ...aus HatReD-TRAIN-Split:     {len(in_train)}")
print(f"  ...aus HatReD-TEST-Split:      {len(in_test)}")
print("-" * 60)
# Leakage-Hinweis: Treffer aus HatReD-train, die bei dir hate sind
leak = in_train & hate_in_dev
print(f"⚠️  LEAKAGE-CHECK:")
print(f"  Deine dev-hate-Memes, die in HatReD-TRAIN liegen: {len(leak)}")
print(f"  -> Würdest du auf HatReD-train fine-tunen und auf")
print(f"     diesem dev evaluieren, wären {len(leak)} Memes geleakt.")
print("=" * 60)

# --- Merge schreiben ---
n_written = 0
with open(OUT, "w", encoding="utf-8") as f:
    for r in dev:
        rid = int(r["id"])
        h = hatred_train.get(rid) or hatred_test.get(rid)
        merged = dict(r)
        if h:
            merged["hatred_target"] = h.get("target")
            merged["hatred_reasoning"] = h.get("reasonings")
            merged["hatred_split"] = "train" if rid in hatred_train else "test"
        if rid in race_all:
            merged["fairface_race"] = race_all[rid].get("race")
        f.write(json.dumps(merged, ensure_ascii=False) + "\n")
        if h:
            n_written += 1

print(f"\nGeschrieben: {OUT}")
print(f"  {n_written} Zeilen mit HatReD-Reasoning angereichert.")

# --- 3 Beispiele zeigen ---
print("\n=== 3 Beispiele (mit Reasoning + race) ===")
shown = 0
for r in dev:
    rid = int(r["id"])
    h = hatred_train.get(rid) or hatred_test.get(rid)
    if h and rid in race_all:
        print(f"\nID {rid} | label={r.get('label')} | text: {r.get('text','')[:50]}")
        print(f"  race (FairFace): {race_all[rid].get('race')}")
        print(f"  target:          {h.get('target')}")
        print(f"  reasoning:       {h.get('reasonings')}")
        shown += 1
        if shown >= 3:
            break

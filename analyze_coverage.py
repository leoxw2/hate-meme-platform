"""Coverage- + Leakage-Analyse: HatReD-Reasoning gegen train/dev/test."""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\Leopo\Claude_Projekte\hate_meme_platform"
HATRED = os.path.join(BASE, "HatRed", "datasets", "hatred", "annotations")
DATA = os.path.join(BASE, "data")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ids(rows):
    return {int(r["id"]) for r in rows}


def hate_ids(rows):
    return {int(r["id"]) for r in rows if r.get("label") == 1}


train = load(os.path.join(DATA, "train.jsonl"))
dev = load(os.path.join(DATA, "dev.jsonl"))
test = load(os.path.join(DATA, "test.jsonl"))

ht = ids(load(os.path.join(HATRED, "fhm_train_reasonings.jsonl")))   # HatReD train
hte = ids(load(os.path.join(HATRED, "fhm_test_reasonings.jsonl")))   # HatReD test
hatred_all = ht | hte

print("=" * 64)
print("DEINE SPLITS")
print(f"  train: {len(train):5d}  (hate: {len(hate_ids(train))})")
print(f"  dev:   {len(dev):5d}  (hate: {len(hate_ids(dev))})")
print(f"  test:  {len(test):5d}  (hate: {len(hate_ids(test))} -- test oft ohne Label)")
print(f"HatReD: train={len(ht)}  test={len(hte)}  gesamt={len(hatred_all)}")
print("=" * 64)

# --- Coverage: wie viele TRAIN-hate-Memes haben ein HatReD-Reasoning? ---
train_hate = hate_ids(train)
cov_train = train_hate & hatred_all
cov_from_ht = train_hate & ht
cov_from_hte = train_hate & hte
print("\nFINE-TUNING-COVERAGE (auf deinem train)")
print(f"  hate-Memes in train:                 {len(train_hate)}")
print(f"  davon mit HatReD-Reasoning:          {len(cov_train)}")
print(f"     aus HatReD-train-Split:           {len(cov_from_ht)}")
print(f"     aus HatReD-test-Split:            {len(cov_from_hte)}")

# --- Leakage: HatReD-Treffer im train, die auch in dev/test liegen ---
print("\nLEAKAGE-CHECK")
print(f"  train ∩ dev (IDs):                   {len(ids(train) & ids(dev))}")
print(f"  train ∩ test (IDs):                  {len(ids(train) & ids(test))}")
print(f"  dev ∩ test (IDs):                    {len(ids(dev) & ids(test))}")
dev_cov = hate_ids(dev) & hatred_all
print(f"  dev-hate mit HatReD-Reasoning:       {len(dev_cov)}  (= Eval-Set, NICHT trainieren)")
overlap = cov_train & ids(dev)
print(f"  train-Reasoning-IDs, die in dev sind:{len(overlap)}  <- muss 0 sein")
print("=" * 64)

# Hateful Memes Web-App — Implementierungsplan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit-basierte Web-App mit vier Tabs zur Steuerung und Live-Inspektion der zweiphasigen Hate-Speech-Klassifizierungspipeline (QWEN → PHI-4-MINI) mit RAG, Fine-Tuning und Experiment-Runner.

**Architecture:** CSV ist die einzige Persistenzschicht während eines Laufs (kein separater Checkpoint-JSON). Resume = CSV-IDs einlesen. Frischer Start = CSV löschen. Ollama-Calls laufen über eine gemeinsame `call_ollama()` Funktion. JSONL-Laden über `load_jsonl()` Utility.

**Tech Stack:** Python 3.10+, Streamlit, pandas, openpyxl, ollama, chromadb, sentence-transformers, scikit-learn

**Changelog v3 (zweiter Opus-Review):**
- Checkpoint-JSON komplett entfernt — CSV ist der Checkpoint
- `ollama_utils.py` mit gemeinsamer `call_ollama()` für QWEN und PHI-4-MINI
- `utils.py` mit `load_jsonl()` — eliminiert dreifach-duplizierten Code
- CSV wird bei frischem Start gelöscht, bei Erfolg ebenfalls
- `_parse_json_response` Loop-Ansatz, kein doppelter Extraktionsblock
- Format-Instruktion gehört in die Prompt-Excel, nicht in `_build_prompt`
- phase2 iteriert direkt über gefilterte Phase-1-Zeilen (kein redundantes Dict)
- Known limitation dokumentiert: `_parse_json_response` Regex bricht bei verschachteltem JSON

---

## Dateistruktur

```
hateful_memes_app/
├── app.py
├── config.py
├── utils.py              ← load_jsonl() und andere gemeinsame Hilfsfunktionen
├── ollama_utils.py       ← call_ollama() für QWEN und PHI-4-MINI
├── excel_utils.py        ← Excel/CSV lesen, schreiben, safe_sheet_name
├── metrics.py
├── rag.py
├── phase1.py
├── phase2.py
├── experiment_runner.py
├── setup_rag.py
├── finetune.py
├── requirements.txt
└── tests/
    ├── test_config.py
    ├── test_metrics.py
    ├── test_excel_utils.py
    ├── test_rag.py
    ├── test_utils.py
    └── test_phase_logic.py
```

---

## Task 1: Projektstruktur und Abhängigkeiten

**Files:**
- Create: `hateful_memes_app/requirements.txt`
- Create: Ordnerstruktur

- [ ] **Schritt 1: Ordner anlegen**

```bash
mkdir -p hateful_memes_app/tests
cd hateful_memes_app
touch app.py config.py utils.py ollama_utils.py excel_utils.py metrics.py rag.py phase1.py phase2.py experiment_runner.py setup_rag.py finetune.py
touch tests/__init__.py tests/test_config.py tests/test_metrics.py tests/test_excel_utils.py tests/test_rag.py tests/test_utils.py tests/test_phase_logic.py
```

- [ ] **Schritt 2: requirements.txt erstellen**

```
streamlit>=1.35.0
pandas>=2.0.0
openpyxl>=3.1.0
ollama>=0.2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
scikit-learn>=1.4.0
Pillow>=10.0.0
pytest>=8.0.0
```

- [ ] **Schritt 3: Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

- [ ] **Schritt 4: Ollama installieren und Modelle pullen**

```bash
# https://ollama.com/download
ollama pull qwen2.5-vl:3b
ollama pull phi4-mini
```

- [ ] **Schritt 5: Commit**

```bash
git init && git add . && git commit -m "chore: project structure and requirements"
```

---

## Task 2: config.py

**Files:**
- Create: `hateful_memes_app/config.py`
- Test: `hateful_memes_app/tests/test_config.py`

- [ ] **Schritt 1: Failing Tests schreiben**

```python
# tests/test_config.py
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
```

- [ ] **Schritt 2: Test ausführen → FAILED**

```bash
pytest tests/test_config.py -v
```

- [ ] **Schritt 3: config.py implementieren**

```python
# config.py
import json, os

DEFAULT_CONFIG = {
    "prompt_excel": "",
    "img_folder": "",
    "results_folder": "",
    "max_tokens_phase1": 2500,
    "max_time_seconds": 120,
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
    """cfg ist Pflichtargument — kein Default, verhindert stille Fehler."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
```

- [ ] **Schritt 4: Tests → 3x PASSED**

```bash
pytest tests/test_config.py -v
```

- [ ] **Schritt 5: Commit**

```bash
git add config.py tests/test_config.py && git commit -m "feat: config load/save"
```

---

## Task 3: utils.py — gemeinsame Hilfsfunktionen

**Files:**
- Create: `hateful_memes_app/utils.py`
- Test: `hateful_memes_app/tests/test_utils.py`

- [ ] **Schritt 1: Failing Tests schreiben**

```python
# tests/test_utils.py
import json, tempfile, os
from utils import load_jsonl

def test_load_jsonl_basic():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.jsonl")
        with open(path, "w") as f:
            f.write('{"id": 1, "text": "hello"}\n')
            f.write('\n')                        # leere Zeile überspringen
            f.write('{"id": 2, "text": "world"}\n')
        result = load_jsonl(path)
        assert len(result) == 2
        assert result[0]["id"] == 1

def test_load_jsonl_empty_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty.jsonl")
        open(path, "w").close()
        assert load_jsonl(path) == []
```

- [ ] **Schritt 2: Test → FAILED**

```bash
pytest tests/test_utils.py -v
```

- [ ] **Schritt 3: utils.py implementieren**

```python
# utils.py
import json

def load_jsonl(path: str) -> list[dict]:
    """Liest eine JSONL-Datei ein. Leere Zeilen werden übersprungen."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
```

- [ ] **Schritt 4: Tests → 2x PASSED**

```bash
pytest tests/test_utils.py -v
```

- [ ] **Schritt 5: Commit**

```bash
git add utils.py tests/test_utils.py && git commit -m "feat: load_jsonl utility"
```

---

## Task 4: ollama_utils.py — gemeinsamer Ollama-Client

**Files:**
- Create: `hateful_memes_app/ollama_utils.py`

- [ ] **Schritt 1: ollama_utils.py implementieren**

```python
# ollama_utils.py
import ollama

def call_ollama(model: str, prompt: str, timeout_secs: int,
                num_predict: int, images: list[str] | None = None) -> tuple[str, str]:
    """Einheitlicher Ollama-Call für QWEN und PHI-4-MINI.

    Args:
        model: Modellname, z.B. "qwen2.5-vl:3b" oder "phi4-mini"
        prompt: Der Prompt-Text
        timeout_secs: Sekunden bis Timeout
        num_predict: Maximale Tokens in der Antwort
        images: Liste von base64-kodierten Bilddaten (nur für QWEN)

    Returns:
        (response_text, status) — status ist "ok", "timeout" oder "error: ..."
    """
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images

    try:
        client = ollama.Client(timeout=timeout_secs)
        resp = client.chat(
            model=model,
            messages=[msg],
            options={"num_predict": num_predict},
        )
        return resp["message"]["content"], "ok"
    except Exception as e:
        err = str(e).lower()
        status = "timeout" if ("time" in err or "timed out" in err) else f"error: {e}"
        return "", status
```

- [ ] **Schritt 2: Manuell testen (Ollama muss laufen)**

```bash
python3 -c "
from ollama_utils import call_ollama
text, status = call_ollama('phi4-mini', 'Sag Hallo.', timeout_secs=30, num_predict=50)
print(status, text[:100])
"
```

Erwartete Ausgabe: `ok Hallo...`

- [ ] **Schritt 3: Commit**

```bash
git add ollama_utils.py && git commit -m "feat: shared ollama call utility"
```

---

## Task 5: excel_utils.py

**Files:**
- Create: `hateful_memes_app/excel_utils.py`
- Test: `hateful_memes_app/tests/test_excel_utils.py`

- [ ] **Schritt 1: Failing Tests schreiben**

```python
# tests/test_excel_utils.py
import os, tempfile, pandas as pd
from excel_utils import write_sheet, read_sheet, get_sheet_names, read_prompts, safe_sheet_name, csv_to_excel_sheet, append_to_csv

def test_write_and_read_sheet():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.xlsx")
        df = pd.DataFrame({"id": [1, 2], "text": ["a", "b"]})
        write_sheet(path, "S1", df)
        assert list(read_sheet(path, "S1")["id"]) == [1, 2]

def test_write_multiple_sheets_preserved():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.xlsx")
        write_sheet(path, "A", pd.DataFrame({"x": [1]}))
        write_sheet(path, "B", pd.DataFrame({"x": [2]}))
        names = get_sheet_names(path)
        assert "A" in names and "B" in names

def test_read_prompts():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.xlsx")
        write_sheet(path, "Phase1", pd.DataFrame(
            {"Name": ["ZS", "CoT"], "Prompt": ["desc1", "desc2"]}))
        p = read_prompts(path, "Phase1")
        assert p["ZS"] == "desc1"

def test_safe_sheet_name_max_31():
    assert len(safe_sheet_name("METRICS_ZS+RP+CoT+ADxCoT+FS+AD+RAG")) <= 31
    assert "×" not in safe_sheet_name("A×B")

def test_append_to_csv_and_csv_to_excel():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "data.csv")
        xlsx_path = os.path.join(d, "data.xlsx")
        append_to_csv({"id": 1, "val": "a"}, csv_path)
        append_to_csv({"id": 2, "val": "b"}, csv_path)
        csv_to_excel_sheet(csv_path, xlsx_path, "Results")
        df = read_sheet(xlsx_path, "Results")
        assert list(df["id"]) == [1, 2]
```

- [ ] **Schritt 2: Test → FAILED**

```bash
pytest tests/test_excel_utils.py -v
```

- [ ] **Schritt 3: excel_utils.py implementieren**

```python
# excel_utils.py
import csv, os
import pandas as pd
from openpyxl import load_workbook

def safe_sheet_name(name: str) -> str:
    """Excel-Sheet-Namen: max 31 Zeichen, keine verbotenen Sonderzeichen."""
    for ch in ["×", "/", "\\", "[", "]", ":", "*", "?", "'"]:
        name = name.replace(ch, "x")
    return name[:31]

def write_sheet(filepath: str, sheet_name: str, df: pd.DataFrame) -> None:
    sheet_name = safe_sheet_name(sheet_name)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    mode = "a" if os.path.exists(filepath) else "w"
    kw = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(filepath, engine="openpyxl", mode=mode, **kw) as w:
        df.to_excel(w, sheet_name=sheet_name, index=False)

def read_sheet(filepath: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")

def get_sheet_names(filepath: str) -> list[str]:
    if not os.path.exists(filepath):
        return []
    wb = load_workbook(filepath, read_only=True)
    names = wb.sheetnames
    wb.close()
    return names

def read_prompts(filepath: str, sheet: str) -> dict[str, str]:
    df = read_sheet(filepath, sheet)
    return dict(zip(df["Name"].astype(str), df["Prompt"].astype(str)))

def append_to_csv(row: dict, csv_path: str) -> None:
    """Hängt eine Zeile an CSV an. Erstellt Datei mit Header wenn nötig."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def csv_to_excel_sheet(csv_path: str, excel_path: str, sheet_name: str) -> None:
    write_sheet(excel_path, sheet_name, pd.read_csv(csv_path, encoding="utf-8"))
```

- [ ] **Schritt 4: Tests → 5x PASSED**

```bash
pytest tests/test_excel_utils.py -v
```

- [ ] **Schritt 5: Commit**

```bash
git add excel_utils.py tests/test_excel_utils.py && git commit -m "feat: excel/csv utilities"
```

---

## Task 6: metrics.py

**Files:**
- Create: `hateful_memes_app/metrics.py`
- Test: `hateful_memes_app/tests/test_metrics.py`

- [ ] **Schritt 1: Failing Tests**

```python
# tests/test_metrics.py
import os, tempfile
from metrics import calculate_metrics, save_metrics_to_excel

def test_perfect_metrics():
    m = calculate_metrics([0,0,1,1], [0,0,1,1], [10,20,80,90])
    assert m["accuracy"] == 1.0 and m["auroc"] == 1.0

def test_auroc_single_class_no_crash():
    m = calculate_metrics([0,0,0], [0,0,1], [10,20,80])
    assert m["auroc"] == "n/a"

def test_save_writes_sheet():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.xlsx")
        save_metrics_to_excel(path, "Test", {"accuracy": 0.6, "auroc": 0.7,
            "precision": 0.5, "recall": 0.8, "f1": 0.6,
            "tp": 10, "fp": 5, "fn": 3, "tn": 20, "n_samples": 38})
        import pandas as pd
        df = pd.read_excel(path, sheet_name="Test", engine="openpyxl")
        assert "accuracy" in df.columns
```

- [ ] **Schritt 2: Test → FAILED**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Schritt 3: metrics.py implementieren**

```python
# metrics.py
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from excel_utils import write_sheet

def calculate_metrics(y_true: list, y_pred: list, y_prob: list) -> dict:
    """Berechnet Klassifizierungsmetriken.
    y_prob: Integer 0-100 (vom Prompt erzwungen), wird zu 0-1 normiert.
    """
    y_prob_norm = [p / 100.0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auroc = round(roc_auc_score(y_true, y_prob_norm), 4)
    except ValueError:
        auroc = "n/a"
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "auroc":     auroc,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_samples": len(y_true),
    }

def save_metrics_to_excel(filepath: str, sheet_name: str, metrics: dict) -> None:
    write_sheet(filepath, sheet_name, pd.DataFrame([metrics]))
```

- [ ] **Schritt 4: Tests → 3x PASSED**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Schritt 5: Commit**

```bash
git add metrics.py tests/test_metrics.py && git commit -m "feat: metrics with AUROC safeguard"
```

---

## Task 7: rag.py

**Files:**
- Create: `hateful_memes_app/rag.py`
- Test: `hateful_memes_app/tests/test_rag.py`

- [ ] **Schritt 1: Failing Tests**

```python
# tests/test_rag.py
import tempfile
from rag import RagRetriever

def test_add_and_query():
    with tempfile.TemporaryDirectory() as d:
        r = RagRetriever(d)
        r.add_documents(["Hate speech targets race.", "Religious discrimination.", "Gender stereotypes."])
        results = r.get_context("racist content", n_results=2)
        assert len(results) == 2 and isinstance(results[0], str)

def test_add_idempotent():
    with tempfile.TemporaryDirectory() as d:
        r = RagRetriever(d)
        r.add_documents(["Same doc."])
        r.add_documents(["Same doc."])
        assert r.count() == 1

def test_empty_returns_empty():
    with tempfile.TemporaryDirectory() as d:
        assert RagRetriever(d).get_context("anything") == []
```

- [ ] **Schritt 2: Test → FAILED**

```bash
pytest tests/test_rag.py -v
```

- [ ] **Schritt 3: rag.py implementieren**

```python
# rag.py
import hashlib
import chromadb
from chromadb.utils import embedding_functions

class RagRetriever:
    COLLECTION = "hate_speech_knowledge"

    def __init__(self, db_path: str):
        self._client = chromadb.PersistentClient(path=db_path)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2")
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION, embedding_function=self._ef)

    def _id(self, doc: str) -> str:
        return hashlib.md5(doc.encode()).hexdigest()[:16]

    def add_documents(self, docs: list[str]) -> None:
        existing = set(self._col.get()["ids"])
        new = [(d, self._id(d)) for d in docs if self._id(d) not in existing]
        if new:
            self._col.add(documents=[d for d, _ in new], ids=[i for _, i in new])

    def get_context(self, query: str, n_results: int = 3) -> list[str]:
        if self._col.count() == 0:
            return []
        n = min(n_results, self._col.count())
        return self._col.query(query_texts=[query], n_results=n)["documents"][0]

    def count(self) -> int:
        return self._col.count()
```

- [ ] **Schritt 4: Tests → 3x PASSED**

```bash
pytest tests/test_rag.py -v
```

- [ ] **Schritt 5: Commit**

```bash
git add rag.py tests/test_rag.py && git commit -m "feat: RAG with content-hash IDs"
```

---

## Task 8: Tests für Kernlogik

**Files:**
- Test: `hateful_memes_app/tests/test_phase_logic.py`

- [ ] **Schritt 1: Tests schreiben**

```python
# tests/test_phase_logic.py

# ── _parse_json_response ────────────────────────────────────────────────────
def test_parse_valid_json():
    from phase2 import _parse_json_response
    raw = '{"label": 1, "confidence": 85, "reasoning": "Racist content."}'
    label, conf, reasoning = _parse_json_response(raw)
    assert label == 1 and conf == 85 and "Racist" in reasoning

def test_parse_json_in_text():
    from phase2 import _parse_json_response
    raw = 'My answer: {"label": 0, "confidence": 70, "reasoning": "No hate."}'
    label, conf, _ = _parse_json_response(raw)
    assert label == 0 and conf == 70

def test_parse_invalid_returns_minus_one():
    from phase2 import _parse_json_response
    label, _, _ = _parse_json_response("Not JSON at all.")
    assert label == -1

# Known limitation: Regex bricht bei verschachteltem JSON
# z.B. {"reasoning": "see {example}", "label": 1} → wird falsch geparst
# Akzeptables Verhalten: gibt -1 zurück, Eintrag wird als parse_error markiert

# ── CSV als Checkpoint ──────────────────────────────────────────────────────
def test_csv_resume_loads_existing_ids():
    import tempfile, os, pandas as pd
    from excel_utils import append_to_csv

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "phase1_ZS.csv")
        append_to_csv({"id": 100, "description": "x", "status": "ok"}, csv_path)
        append_to_csv({"id": 200, "description": "y", "status": "ok"}, csv_path)

        # Simuliere Resume-Logik aus phase1.run_phase1
        processed_ids = set(pd.read_csv(csv_path)["id"].astype(int))
        assert 100 in processed_ids
        assert 200 in processed_ids
        assert len(processed_ids) == 2
```

- [ ] **Schritt 2: Test → FAILED (phase2 noch leer)**

```bash
pytest tests/test_phase_logic.py -v
```

- [ ] **Schritt 3: Commit**

```bash
git add tests/test_phase_logic.py && git commit -m "test: phase logic tests"
```

---

## Task 9: phase1.py

**Files:**
- Create: `hateful_memes_app/phase1.py`

- [ ] **Schritt 1: phase1.py implementieren**

```python
# phase1.py
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
                model="qwen2.5-vl:3b",
                prompt=prompt_text,
                timeout_secs=max_time_secs,
                num_predict=max_tokens,
                images=[img_b64],
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
```

- [ ] **Schritt 2: CSV-Checkpoint-Test ausführen**

```bash
pytest tests/test_phase_logic.py::test_csv_resume_loads_existing_ids -v
```

Erwartete Ausgabe: `PASSED`.

- [ ] **Schritt 3: Commit**

```bash
git add phase1.py && git commit -m "feat: phase1 QWEN pipeline, CSV as checkpoint"
```

---

## Task 10: phase2.py

**Files:**
- Create: `hateful_memes_app/phase2.py`

- [ ] **Schritt 1: phase2.py implementieren**

```python
# phase2.py
import json, os, re
import pandas as pd
from utils import load_jsonl
from ollama_utils import call_ollama
from excel_utils import (append_to_csv, csv_to_excel_sheet, get_sheet_names,
                         read_sheet, safe_sheet_name)
from metrics import calculate_metrics, save_metrics_to_excel

# Known limitation: Regex `_first_brace_block` bricht bei verschachteltem JSON
# z.B. {"reasoning": "see {example}", "label": 1} → gibt -1 zurück (parse_error)
# Das ist akzeptabel: Eintrag wird übersprungen, Lauf läuft weiter.
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

    WICHTIG: phase1_sheet muss mit dev.jsonl (Experiment-Modus) generiert sein,
    nicht mit train.jsonl — sonst ist der ID-Overlap mit dev.jsonl leer.

    Format-Hinweis: Der Prompt in prompt_text MUSS die JSON-Format-Instruktion
    enthalten (confidence als Integer 0-100). Nicht im Code erzwungen.

    Yielded: progress, log, done (mit metrics dict)
    """
    # Phase-1-Daten direkt iterieren — kein redundantes Dict
    phase1_df = read_sheet(phase1_excel, phase1_sheet)
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
        processed_ids = set(pd.read_csv(csv_p)["id"].astype(int))
        # Bereits verarbeitete Ergebnisse für Metriken nachladen
        existing = pd.read_csv(csv_p)
        y_true = list(existing.loc[
            (existing["status"] == "ok") & (existing["pred_label"].isin([0, 1])),
            "true_label"].astype(int))
        y_pred = list(existing.loc[
            (existing["status"] == "ok") & (existing["pred_label"].isin([0, 1])),
            "pred_label"].astype(int))
        y_prob = list(existing.loc[
            (existing["status"] == "ok") & (existing["pred_label"].isin([0, 1])),
            "confidence"].astype(float))
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

        # Prompt zusammenbauen: RAG-Kontext vor Bildbeschreibung
        context_block = ""
        if rag_context:
            context_block = "\n\nRelevantes Hintergrundwissen:\n" + "\n".join(
                f"- {c}" for c in rag_context)
        final_prompt = (f"{prompt_text}{context_block}\n\n"
                        f"Bildbeschreibung: {description}\n"
                        f"Text auf dem Bild: {meme_text}")

        raw, call_status = call_ollama(
            model=model_name, prompt=final_prompt,
            timeout_secs=60, num_predict=300)

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
        save_metrics_to_excel(phase2_excel, safe_sheet_name(f"M_{sheet_name}"), metrics)

    yield {"type": "done", "metrics": metrics}
```

- [ ] **Schritt 2: Phase-Logic-Tests ausführen**

```bash
pytest tests/test_phase_logic.py -v
```

Erwartete Ausgabe: 4x `PASSED`.

- [ ] **Schritt 3: Commit**

```bash
git add phase2.py && git commit -m "feat: phase2 with CSV checkpoint, no separate JSON, resume-safe metrics"
```

---

## Task 11: experiment_runner.py

**Files:**
- Create: `hateful_memes_app/experiment_runner.py`

- [ ] **Schritt 1: experiment_runner.py implementieren**

```python
# experiment_runner.py
import json, os
import pandas as pd
from phase2 import run_phase2
from excel_utils import safe_sheet_name

PHASE2_CONFIG = {
    "ZS":            ("ZS",        False, False),
    "CoT+FS+AD":     ("CoT+FS+AD", False, False),
    "CoT+FS+AD+RAG": ("CoT+FS+AD", True,  False),
    "CoT+FS+AD+FT":  ("CoT+FS+AD", False, True),
    "ZS+RAG":        ("ZS",        True,  False),
    "ZS+FT":         ("ZS",        False, True),
}

def _runner_csv(results_folder: str) -> str:
    return os.path.join(results_folder, "runner_completed.csv")

def run_experiments(phase1_selections: list[str], phase2_selections: list[str],
                    prompts_phase2: dict[str, str],
                    phase1_excel: str, dev_jsonl_path: str,
                    phase2_excel: str, results_folder: str,
                    ft_model_path: str = "", rag_retriever=None):
    """Generator: alle ausgewählten Phase1 × Phase2 Kombinationen.
    Checkpoint: CSV mit abgeschlossenen Kombinations-Keys.
    """
    combinations = [(p1, p2)
                    for p1 in phase1_selections
                    for p2 in phase2_selections]
    total = len(combinations)

    # Abgeschlossene Kombinationen aus CSV laden
    runner_csv = _runner_csv(results_folder)
    completed = set()
    if os.path.exists(runner_csv):
        completed = set(pd.read_csv(runner_csv)["combo_key"].astype(str))

    for i, (phase1_sheet, phase2_name) in enumerate(combinations):
        combo_key = safe_sheet_name(f"{phase1_sheet}x{phase2_name}")
        if combo_key in completed:
            continue

        base_prompt_key, use_rag, use_ft = PHASE2_CONFIG[phase2_name]
        prompt_text = prompts_phase2.get(base_prompt_key, "")

        yield {"type": "combination_start", "phase1": phase1_sheet,
               "phase2": phase2_name, "index": i + 1, "total": total}

        metrics = {}
        for update in run_phase2(
            phase1_excel=phase1_excel,
            phase1_sheet=phase1_sheet,
            dev_jsonl_path=dev_jsonl_path,
            phase2_excel=phase2_excel,
            prompt_name=phase2_name,
            prompt_text=prompt_text,
            use_rag=use_rag,
            use_ft=use_ft,
            results_folder=results_folder,
            ft_model_path=ft_model_path,
            rag_retriever=rag_retriever if use_rag else None,
        ):
            if update["type"] == "done":
                metrics = update.get("metrics", {})
            else:
                yield update

        # Abgeschlossene Kombination in CSV speichern
        from excel_utils import append_to_csv
        append_to_csv({"combo_key": combo_key}, runner_csv)
        completed.add(combo_key)

        yield {"type": "combination_done", "phase1": phase1_sheet,
               "phase2": phase2_name, "metrics": metrics}

    yield {"type": "all_done", "completed": len(completed)}
```

- [ ] **Schritt 2: Commit**

```bash
git add experiment_runner.py && git commit -m "feat: experiment runner with CSV-based checkpoint"
```

---

## Task 12: app.py — alle vier Tabs

**Files:**
- Create: `hateful_memes_app/app.py`

- [ ] **Schritt 1: app.py implementieren**

```python
# app.py
import os
import streamlit as st
from config import load_config, save_config, CONFIG_PATH

st.set_page_config(page_title="Hateful Memes Pipeline", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()

def _render_log(entries: list, container) -> None:
    with container:
        st.markdown("\n".join(entries[:50]))

tab_s, tab_p1, tab_p2, tab_r = st.tabs(
    ["⚙️ Einstellungen", "📷 Phase 1", "🧠 Phase 2", "🚀 Experiment-Runner"])

# ── EINSTELLUNGEN ─────────────────────────────────────────────────────────────
with tab_s:
    st.header("Einstellungen")
    cfg = st.session_state.cfg

    cfg["prompt_excel"]   = st.text_input("Prompt-Excel", value=cfg.get("prompt_excel",""))
    cfg["img_folder"]     = st.text_input("Bild-Ordner (img/)", value=cfg.get("img_folder",""))
    cfg["results_folder"] = st.text_input("Ergebnis-Ordner", value=cfg.get("results_folder",""))

    c1, c2 = st.columns(2)
    with c1:
        cfg["max_tokens_phase1"] = st.number_input(
            "Max. Token pro Bildbeschreibung", 100, 10000,
            value=cfg.get("max_tokens_phase1", 2500), step=100)
    with c2:
        cfg["max_time_seconds"] = st.number_input(
            "Max. Zeit pro Bild (Sek.)", 10, 600,
            value=cfg.get("max_time_seconds", 120), step=10)

    cfg["phase1_excel"]  = st.text_input("Phase-1-Ergebnis-Excel", value=cfg.get("phase1_excel",""))
    cfg["phase2_excel"]  = st.text_input("Phase-2-Ergebnis-Excel", value=cfg.get("phase2_excel",""))
    cfg["ft_model_path"] = st.text_input(
        "Fine-Tuned Modell (leer lassen bis FT fertig)",
        value=cfg.get("ft_model_path",""))

    if st.button("💾 Speichern"):
        save_config(CONFIG_PATH, cfg)
        st.session_state.cfg = cfg
        st.success("Gespeichert.")

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
with tab_p1:
    st.header("Phase 1 — Bildbeschreibungen")
    cfg = st.session_state.cfg

    prompts_p1 = {}
    if cfg.get("prompt_excel") and os.path.exists(cfg["prompt_excel"]):
        from excel_utils import read_prompts
        try:
            prompts_p1 = read_prompts(cfg["prompt_excel"], "Phase1")
        except Exception as e:
            st.warning(f"Prompt-Excel: {e}")

    if not prompts_p1:
        st.info("Bitte Prompt-Excel in Einstellungen hinterlegen.")
    else:
        mode_p1 = st.radio("Modus",
            ["Experiment-Modus (dev.jsonl)", "Fine-Tuning-Modus (train.jsonl)"],
            horizontal=True,
            help="Experiment: ~500 Bilder für Phase-2-Auswertung. FT: 8500 Bilder für QLoRA.")
        jsonl_name = "dev.jsonl" if "Experiment" in mode_p1 else "train.jsonl"
        jsonl_path = os.path.join(os.path.dirname(cfg.get("img_folder","")), jsonl_name)

        prompt_name = st.selectbox("Prompt", list(prompts_p1.keys()))
        prompt_text = st.text_area("Prompt-Text", value=prompts_p1[prompt_name], height=150)

        from phase1 import get_run_info, clear_run
        run_info = get_run_info(prompt_name, cfg.get("results_folder",""))
        resume = False
        if run_info:
            ca, cb = st.columns([3, 1])
            with ca:
                st.info(f"Unterbrochener Lauf: {run_info['n_processed']} Einträge bereits verarbeitet")
            with cb:
                resume = st.checkbox("Fortsetzen", value=True)
                if st.button("Neu starten"):
                    clear_run(prompt_name, cfg["results_folder"])
                    st.rerun()

        if st.button("▶️ Phase 1 starten"):
            if not cfg.get("img_folder") or not cfg.get("phase1_excel"):
                st.error("Bitte Bild-Ordner und Phase-1-Excel angeben.")
            elif not os.path.exists(jsonl_path):
                st.error(f"{jsonl_name} nicht gefunden: {jsonl_path}")
            else:
                from phase1 import run_phase1
                bar = st.progress(0.0)
                log_box = st.container()
                log_entries = []
                for upd in run_phase1(
                    jsonl_path=jsonl_path, img_folder=cfg["img_folder"],
                    phase1_excel=cfg["phase1_excel"], prompt_name=prompt_name,
                    prompt_text=prompt_text, max_tokens=cfg["max_tokens_phase1"],
                    max_time_secs=cfg["max_time_seconds"],
                    results_folder=cfg["results_folder"], resume=resume):
                    if upd["type"] == "progress":
                        bar.progress(upd["current"] / max(upd["total"], 1),
                                     text=f"{upd['current']} / {upd['total']}")
                    elif upd["type"] == "log":
                        icon = "✅" if upd["status"] == "ok" else "⚠️"
                        log_entries.insert(0,
                            f"{icon} **ID {upd['id']}** | _{upd['text'][:60]}_\n\n"
                            f"{upd['description']}\n\n---")
                        _render_log(log_entries, log_box)
                    elif upd["type"] == "done":
                        st.success(f"Fertig! ✅ {upd['total_ok']} OK, ⚠️ {upd['total_skip']} übersprungen")

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
with tab_p2:
    st.header("Phase 2 — Klassifizierung")
    cfg = st.session_state.cfg

    prompts_p2 = {}
    if cfg.get("prompt_excel") and os.path.exists(cfg["prompt_excel"]):
        from excel_utils import read_prompts
        try:
            prompts_p2 = read_prompts(cfg["prompt_excel"], "Phase2")
        except Exception as e:
            st.warning(f"Prompt-Excel: {e}")

    p1_sheets = []
    if cfg.get("phase1_excel") and os.path.exists(cfg["phase1_excel"]):
        from excel_utils import get_sheet_names
        p1_sheets = [s for s in get_sheet_names(cfg["phase1_excel"])
                     if not s.startswith("M_")]

    if not p1_sheets:
        st.info("Zuerst Phase 1 im Experiment-Modus ausführen.")
    elif not prompts_p2:
        st.info("Prompt-Excel in Einstellungen hinterlegen.")
    else:
        p1_sheet   = st.selectbox("Phase-1-Sheet", p1_sheets)
        p_name     = st.selectbox("Basis-Prompt", list(prompts_p2.keys()))
        p_text     = st.text_area("Prompt-Text", value=prompts_p2.get(p_name,""), height=200)
        use_rag    = st.checkbox("RAG aktivieren")
        use_ft     = st.checkbox("Fine-Tuned Modell",
                                 disabled=not bool(cfg.get("ft_model_path")),
                                 help="FT-Pfad in Einstellungen eintragen.")

        if st.button("▶️ Single Run"):
            dev_jsonl = os.path.join(os.path.dirname(cfg["img_folder"]), "dev.jsonl")
            rag_r = None
            if use_rag:
                from rag import RagRetriever
                rag_r = RagRetriever(os.path.join(cfg["results_folder"], "chroma_db"))
            bar = st.progress(0.0)
            log_box = st.container()
            log_entries = []
            from phase2 import run_phase2
            for upd in run_phase2(
                phase1_excel=cfg["phase1_excel"], phase1_sheet=p1_sheet,
                dev_jsonl_path=dev_jsonl, phase2_excel=cfg["phase2_excel"],
                prompt_name=p_name, prompt_text=p_text,
                use_rag=use_rag, use_ft=use_ft,
                results_folder=cfg["results_folder"],
                ft_model_path=cfg.get("ft_model_path",""),
                rag_retriever=rag_r):
                if upd["type"] == "progress":
                    bar.progress(upd["current"] / max(upd["total"], 1),
                                 text=f"{upd['current']} / {upd['total']}")
                elif upd["type"] == "log":
                    icon = "🔴" if upd["label"] == 1 else "🟢"
                    log_entries.insert(0,
                        f"{icon} **ID {upd['id']}** | Label: {upd['label']} | Conf: {upd['confidence']:.0f}%\n\n"
                        f"_{upd['text'][:80]}_\n\n**Reasoning:** {upd['reasoning']}\n\n---")
                    _render_log(log_entries, log_box)
                elif upd["type"] == "done":
                    m = upd.get("metrics", {})
                    if m:
                        st.success(f"✅ Acc {m.get('accuracy',0):.1%} | "
                                   f"AUROC {m.get('auroc','n/a')} | "
                                   f"F1 {m.get('f1',0):.3f} | n={m.get('n_samples',0)}")

# ── EXPERIMENT-RUNNER ─────────────────────────────────────────────────────────
with tab_r:
    st.header("Experiment-Runner")
    cfg = st.session_state.cfg

    st.subheader("Phase 1")
    p1_opts = ["ZS", "ZS+RP+AD", "ZS+RP+CoT+AD", "ZS+RP+AD(min)"]
    p1_sel = [n for n in p1_opts if st.checkbox(n, key=f"r1_{n}")]

    st.subheader("Phase 2")
    p2_opts = ["ZS", "CoT+FS+AD", "CoT+FS+AD+RAG", "CoT+FS+AD+FT", "ZS+RAG", "ZS+FT"]
    p2_sel = [n for n in p2_opts if st.checkbox(n, key=f"r2_{n}")]

    n = len(p1_sel) * len(p2_sel)
    st.info(f"**{n} Kombinationen ausgewählt**")

    if st.button("🚀 Starten", disabled=(n == 0)):
        prompts_r = {}
        if cfg.get("prompt_excel"):
            from excel_utils import read_prompts
            try:
                prompts_r = read_prompts(cfg["prompt_excel"], "Phase2")
            except Exception:
                pass
        rag_r = None
        if any("RAG" in p for p in p2_sel):
            from rag import RagRetriever
            rag_r = RagRetriever(os.path.join(cfg["results_folder"], "chroma_db"))
        dev_jsonl = os.path.join(os.path.dirname(cfg["img_folder"]), "dev.jsonl")
        combo_info = st.empty()
        bar_total = st.progress(0.0)
        bar_combo = st.progress(0.0)
        log_box = st.container()
        log_entries = []
        n_done = 0
        from experiment_runner import run_experiments
        for upd in run_experiments(
            phase1_selections=p1_sel, phase2_selections=p2_sel,
            prompts_phase2=prompts_r, phase1_excel=cfg["phase1_excel"],
            dev_jsonl_path=dev_jsonl, phase2_excel=cfg["phase2_excel"],
            results_folder=cfg["results_folder"],
            ft_model_path=cfg.get("ft_model_path",""), rag_retriever=rag_r):
            if upd["type"] == "combination_start":
                combo_info.info(f"Kombination {upd['index']}/{upd['total']}: "
                                f"**{upd['phase1']} × {upd['phase2']}**")
                bar_total.progress(upd["index"] / max(upd["total"], 1))
                bar_combo.progress(0.0)
            elif upd["type"] == "progress":
                bar_combo.progress(upd["current"] / max(upd["total"], 1),
                                   text=f"{upd['current']} / {upd['total']}")
            elif upd["type"] == "log":
                icon = "🔴" if upd["label"] == 1 else "🟢"
                log_entries.insert(0,
                    f"{icon} **ID {upd['id']}** | _{upd['text'][:60]}_\n\n---")
                _render_log(log_entries, log_box)
            elif upd["type"] == "combination_done":
                n_done += 1
                m = upd.get("metrics", {})
                if m:
                    st.success(f"✅ {upd['phase1']} × {upd['phase2']}: "
                               f"Acc {m.get('accuracy',0):.1%} | AUROC {m.get('auroc','n/a')}")
            elif upd["type"] == "all_done":
                st.balloons()
                st.success(f"Alle {n_done} Kombinationen abgeschlossen!")
```

- [ ] **Schritt 2: App starten und alle Tabs testen**

```bash
streamlit run app.py
```

- [ ] **Schritt 3: Commit**

```bash
git add app.py && git commit -m "feat: complete streamlit app v3"
```

---

## Task 13: setup_rag.py

**Files:**
- Create: `hateful_memes_app/setup_rag.py`

- [ ] **Schritt 1: setup_rag.py erstellen**

```python
# setup_rag.py
"""Einmalig ausführen: befüllt ChromaDB.
Dokumente auf Englisch — all-MiniLM-L6-v2 ist primär englisch trainiert.
"""
import os
from config import load_config
from rag import RagRetriever

DOCS = [
    "Hate speech based on race dehumanizes people using slurs or promotes racial superiority.",
    "Religious hate speech mocks or calls for violence against people based on faith.",
    "Gender hate speech uses stereotypes to demean women or LGBTQ+ people.",
    "Hate targeting nationality portrays immigrants as criminals or invaders.",
    "Disability hate speech mocks people with physical or mental disabilities.",
    "Antisemitism includes Holocaust denial, conspiracy theories, dehumanizing portrayals.",
    "Hateful memes combine innocent images with text to create hateful meaning through juxtaposition.",
    "Dog whistles are coded language conveying hateful messages with plausible deniability.",
    "Benign statements become hateful combined with images targeting a protected group.",
    "Humor or irony framing does not exempt content from being hate speech.",
    "Dehumanization compares groups to vermin, parasites, or animals.",
    "Historical enemy comparisons imply a group is dangerous or untrustworthy.",
    "Calls for exclusion, segregation, or violence against a protected group are hate speech.",
    "Stereotypes portraying groups as lazy, criminal, greedy, or deviant are hate speech.",
]

if __name__ == "__main__":
    cfg = load_config()
    db_path = os.path.join(cfg.get("results_folder", "."), "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    r = RagRetriever(db_path)
    r.add_documents(DOCS)
    print(f"ChromaDB: {r.count()} Dokumente in {db_path}")
```

- [ ] **Schritt 2: Ausführen**

```bash
python setup_rag.py
```

Erwartete Ausgabe: `ChromaDB: 14 Dokumente in ...`

- [ ] **Schritt 3: Commit**

```bash
git add setup_rag.py && git commit -m "feat: RAG knowledge base setup"
```

---

## Task 14: Fine-Tuning Integration (nach QLoRA-Training)

Dieser Task wird ausgeführt **nachdem** das QLoRA-Training auf train.jsonl abgeschlossen ist.

- [ ] **Schritt 1: Trainingsdaten vorbereiten**

```python
# finetune.py
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

    # Format-Instruktion im Prompt — konsistent mit Prompt-Excel Phase2
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
```

- [ ] **Schritt 2: QLoRA-Training**

```bash
pip install peft transformers trl accelerate bitsandbytes
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import torch, datasets

model_name = 'microsoft/Phi-4-mini-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_4bit=True, torch_dtype=torch.float16)

lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    target_modules=['q_proj','v_proj'], lora_dropout=0.1)
model = get_peft_model(model, lora)

ds = datasets.load_dataset('json', data_files='results/finetune_data.jsonl')['train']
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(output_dir='results/ft_model', num_train_epochs=3,
                   per_device_train_batch_size=2, save_steps=100))
trainer.train()
model.save_pretrained('results/ft_model')
print('Fertig: results/ft_model')
"
```

Dauert ca. 2-4 Stunden auf 8 GB GPU.

- [ ] **Schritt 3: Modell in Ollama registrieren**

```bash
echo 'FROM phi4-mini
ADAPTER ./results/ft_model' > Modelfile

ollama create phi4-mini-ft -f Modelfile
```

- [ ] **Schritt 4: Pfad in Einstellungen eintragen**

App → Einstellungen → Fine-Tuned Modell: `phi4-mini-ft` eintragen. FT-Checkboxen sind ab jetzt aktiv.

- [ ] **Schritt 5: Commit**

```bash
git add finetune.py && git commit -m "feat: QLoRA fine-tuning preparation and Ollama registration"
```

---

## Task 15: Abschlusstest

- [ ] **Alle Unit-Tests**

```bash
pytest tests/ -v
```

Erwartete Ausgabe: alle `PASSED`.

- [ ] **End-to-End mit kleinem Datensatz**

```bash
python3 -c "
with open('data/dev.jsonl') as f:
    lines = [f.readline() for _ in range(20)]
with open('/tmp/test_dev.jsonl', 'w') as f:
    f.writelines(lines)
print('20 Einträge erstellt')
"
```

Checkliste:
- [ ] Einstellungen speichern → config.json prüfen
- [ ] Phase 1 Experiment-Modus auf 5 Bildern
- [ ] Strg+C → mit Resume fortsetzen → keine Duplikate in CSV
- [ ] Phase 2 Single Run: ZS-Sheet + ZS-Prompt
- [ ] Phase 2 Single Run: RAG aktivieren
- [ ] Experiment-Runner: 2 × 2 Kombinationen

- [ ] **Final Commit**

```bash
git add . && git commit -m "feat: hateful memes pipeline v3 — all review fixes applied"
```

---

## Hinweise für die Live-Demo

1. `ollama serve` im Hintergrund starten
2. `python setup_rag.py` einmalig ausführen
3. Phase 1 Experiment-Modus vorab auf allen dev.jsonl-Bildern abschließen
4. Ein eindeutiges Meme aus dev.jsonl vorher identifizieren (Label=1, klarer Fall)
5. Demo: Phase 2 Single Run mit diesem Meme — Reasoning live im Log zeigen
6. Prompt-Excel muss die JSON-Format-Instruktion in jedem Phase-2-Prompt enthalten

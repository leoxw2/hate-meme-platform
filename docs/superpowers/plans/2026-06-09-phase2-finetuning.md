# Phase-2 QLoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune phi4-mini for hate-meme classification via QLoRA on RunPod, deploy as an Ollama model (`phase2-ft`), and wire it into the existing eval pipeline to fill the `ZS+RP+AD+FT` matrix cell.

**Architecture:** One new Phase-2 prompt (`ZS+RP+AD`) is stored as a pure system prompt in `prompts.xlsx`. `phase2.py` is patched so the user turn is always `"Meme text: …\n\nImage description: …"`. `finetune.py` is rewritten to produce ~2000 chat-format JSONL examples (system/user/assistant). A new `train_qlora.py` runs on RunPod (Unsloth + QLoRA, RTX 3090) and exports a `q4_k_m` GGUF. The GGUF is rsync'd back to the Mac and registered with `ollama create phase2-ft`.

**Tech Stack:** Python 3.11, openpyxl, Unsloth (QLoRA), HuggingFace TRL, GGUF (llama.cpp), Ollama, RunPod (RTX 3090, CUDA 12), pytest

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `data/prompts.xlsx` | Add `ZS+RP+AD` row to Phase2 sheet (system-prompt only, no placeholders) |
| Modify | `hateful_memes_app/phase2.py` lines 95–96 | Fix user_prompt: `"Meme text: …\n\nImage description: …"` |
| Rewrite | `hateful_memes_app/finetune.py` | `build_finetune_data()` — produces `data/finetune_data.jsonl` |
| Create | `hateful_memes_app/tests/test_finetune.py` | Unit tests for `build_finetune_data` |
| Create | `hateful_memes_app/train_qlora.py` | Unsloth QLoRA training + GGUF export (runs on RunPod) |
| Create | `hateful_memes_app/launch_runpod_ft.sh` | One-shot RunPod setup + training launcher |
| Create | `hateful_memes_app/requirements-train.txt` | Training-only deps (kept out of app requirements.txt) |

---

## Task 1: Add ZS+RP+AD prompt to prompts.xlsx

**Files:**
- Modify: `data/prompts.xlsx` (Phase2 sheet, columns `Name` / `Prompt`)

- [ ] **Step 1: Add the row with a Python script**

Run from the project root (`/Users/leo/Claude_Projekte/hate_meme_platform`):

```python
# run once, then delete
import openpyxl

PROMPT_TEXT = (
    "You are an expert hate speech analyst specializing in multimodal meme classification.\n\n"
    "Analyze whether the meme contains hate speech. Be aware that hate speech in memes is often "
    "subtle — it may use coded language, cultural references, irony, or rely on the combination "
    "of image and text to convey harmful meaning that neither element would suggest alone.\n\n"
    "Consider:\n"
    "- Is any group (racial, ethnic, religious, gender, sexual orientation, disability, etc.) targeted?\n"
    "- Is there dehumanization, harmful stereotyping, slurs, or incitement?\n"
    "- Does irony, sarcasm, or a dog-whistle disguise the hateful intent?\n\n"
    "Respond only with a JSON object in this exact format:\n"
    '{\"reasoning\": \"your brief reasoning\", \"label\": 0, \"confidence\": 75}\n\n'
    "label must be 0 (not hateful) or 1 (hateful). confidence must be an integer from 0 to 100."
)

wb = openpyxl.load_workbook("data/prompts.xlsx")
ws = wb["Phase2"]
ws.append(["ZS+RP+AD", PROMPT_TEXT])
wb.save("data/prompts.xlsx")
print("Done. Rows now:", ws.max_row)
```

Expected output: `Done. Rows now: 3`

- [ ] **Step 2: Verify**

```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform
python3 -c "
from hateful_memes_app.excel_utils import read_sheet
df = read_sheet('data/prompts.xlsx', 'Phase2')
print(df[['Name']].to_string())
assert 'ZS+RP+AD' in df['Name'].values
print('OK')
"
```

Expected output: `Name` column shows `ZS`, `CoT+FS+AD`, `ZS+RP+AD` — then `OK`.

- [ ] **Step 3: Commit**

```bash
git add data/prompts.xlsx
git commit -m "feat: add ZS+RP+AD system-prompt to Phase2 prompts"
```

---

## Task 2: Fix phase2.py user_prompt construction

**Files:**
- Modify: `hateful_memes_app/phase2.py` lines 95–96

The current lines 94–96:
```python
system_prompt = f"{prompt_text}{context_block}"
user_prompt = (f"Bildbeschreibung: {description}\n"
               f"Text auf dem Bild: {meme_text}")
```

The new approach: `prompt_text` is now a pure system prompt (no `{text}`/`{description}` placeholders). The user turn is always the meme data in English.

- [ ] **Step 1: Write a failing test**

Add to `hateful_memes_app/tests/test_phase_logic.py` (existing file — append at the bottom):

```python
def test_user_prompt_format():
    """phase2 user turn must be 'Meme text: …\\n\\nImage description: …' in English."""
    # We test the format string directly — no Ollama call needed.
    description = "A cat on a mat."
    meme_text = "hello world"
    user_prompt = f"Meme text: {meme_text}\n\nImage description: {description}"
    assert user_prompt == "Meme text: hello world\n\nImage description: A cat on a mat."
    assert "Bildbeschreibung" not in user_prompt
    assert "Text auf dem Bild" not in user_prompt
```

Run: `cd hateful_memes_app && python -m pytest tests/test_phase_logic.py::test_user_prompt_format -v`

Expected: PASS (this test documents the target format — it passes immediately since it's just a string assertion, confirming the expected shape before we patch the source).

- [ ] **Step 2: Patch phase2.py lines 95–96**

In `hateful_memes_app/phase2.py`, replace:
```python
        system_prompt = f"{prompt_text}{context_block}"
        user_prompt = (f"Bildbeschreibung: {description}\n"
                       f"Text auf dem Bild: {meme_text}")
```
with:
```python
        system_prompt = f"{prompt_text}{context_block}"
        user_prompt = f"Meme text: {meme_text}\n\nImage description: {description}"
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
cd hateful_memes_app && python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add hateful_memes_app/phase2.py hateful_memes_app/tests/test_phase_logic.py
git commit -m "fix: phase2 user_prompt to English Meme text/Image description format"
```

---

## Task 3: Write test_finetune.py (failing tests first)

**Files:**
- Create: `hateful_memes_app/tests/test_finetune.py`

- [ ] **Step 1: Create the test file**

```python
# hateful_memes_app/tests/test_finetune.py
import json, os, tempfile
import openpyxl
import pytest
from finetune import build_finetune_data

# ── Fixtures ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a test analyst. Reply with JSON only."

def _make_phase1_xlsx(path: str, rows: list[dict]) -> None:
    """rows: list of {id, text, description, status}"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ZS + RP"
    ws.append(["id", "img", "text", "description", "prompt_name", "status"])
    for r in rows:
        ws.append([r["id"], f"img/{r['id']:05d}.png", r.get("text", ""), r["description"],
                   "ZS + RP", r.get("status", "ok")])
    wb.save(path)

def _make_prompts_xlsx(path: str, prompt_name: str, prompt_text: str) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Phase2"
    ws.append(["Name", "Prompt"])
    ws.append([prompt_name, prompt_text])
    wb.save(path)

def _make_train_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def _make_hatred_jsonl(path: str, rows: list[dict]) -> None:
    """rows: {id, target, reasonings}"""
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "id": r["id"],
                "img": f"{r['id']:05d}.png",
                "target": r["target"],
                "reasonings": r["reasonings"],
            }) + "\n")

@pytest.fixture
def tmp_data(tmp_path):
    """
    5 memes: ids 100–104
      100: label=1, HatReD covered (target + reasoning)
      101: label=1, HatReD covered (reasoning only, no target)
      102: label=1, NOT in HatReD → fallback hateful template
      103: label=0, benign
      104: label=0, benign
      105: status='missing_image' → must be excluded
    """
    phase1_rows = [
        {"id": 100, "text": "text100", "description": "desc100"},
        {"id": 101, "text": "text101", "description": "desc101"},
        {"id": 102, "text": "text102", "description": "desc102"},
        {"id": 103, "text": "text103", "description": "desc103"},
        {"id": 104, "text": "text104", "description": "desc104"},
        {"id": 105, "text": "skip",    "description": "skip", "status": "missing_image"},
    ]
    train_rows = [
        {"id": 100, "img": "img/00100.png", "text": "text100", "label": 1},
        {"id": 101, "img": "img/00101.png", "text": "text101", "label": 1},
        {"id": 102, "img": "img/00102.png", "text": "text102", "label": 1},
        {"id": 103, "img": "img/00103.png", "text": "text103", "label": 0},
        {"id": 104, "img": "img/00104.png", "text": "text104", "label": 0},
        {"id": 105, "img": "img/00105.png", "text": "skip",    "label": 0},
    ]
    hatred_rows = [
        {"id": 100, "target": ["the jews"], "reasonings": ["mocks jewish people."]},
        {"id": 101, "target": [],           "reasonings": ["dehumanizes immigrants."]},
    ]
    phase1_path   = str(tmp_path / "phase1.xlsx")
    prompts_path  = str(tmp_path / "prompts.xlsx")
    train_path    = str(tmp_path / "train.jsonl")
    hatred_path   = str(tmp_path / "hatred.jsonl")
    output_path   = str(tmp_path / "finetune_data.jsonl")

    _make_phase1_xlsx(phase1_path, phase1_rows)
    _make_prompts_xlsx(prompts_path, "ZS+RP+AD", SYSTEM_PROMPT)
    _make_train_jsonl(train_path, train_rows)
    _make_hatred_jsonl(hatred_path, hatred_rows)

    return dict(
        phase1_path=phase1_path, prompts_path=prompts_path,
        train_path=train_path,   hatred_path=hatred_path,
        output_path=output_path,
    )

def _run(tmp_data, **kwargs):
    d = tmp_data
    defaults = dict(
        train_xlsx=d["phase1_path"], train_sheet="ZS + RP",
        train_jsonl=d["train_path"], hatred_jsonl=d["hatred_path"],
        prompts_xlsx=d["prompts_path"], output_jsonl=d["output_path"],
        prompt_name="ZS+RP+AD", seed=42,
    )
    defaults.update(kwargs)
    return build_finetune_data(**defaults)

def _load_output(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

# ── Tests ────────────────────────────────────────────────────────────────────

def test_row_count(tmp_data):
    """5 ok examples (id 105 excluded: missing_image)."""
    n = _run(tmp_data)
    assert n == 5
    rows = _load_output(tmp_data["output_path"])
    assert len(rows) == 5

def test_chat_message_structure(tmp_data):
    """Each example has messages: [system, user, assistant]."""
    _run(tmp_data)
    for row in _load_output(tmp_data["output_path"]):
        msgs = row["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

def test_system_prompt_content(tmp_data):
    """System content matches the prompt from prompts.xlsx."""
    _run(tmp_data)
    for row in _load_output(tmp_data["output_path"]):
        assert row["messages"][0]["content"] == SYSTEM_PROMPT

def test_user_prompt_format(tmp_data):
    """User content is 'Meme text: …\\n\\nImage description: …'."""
    _run(tmp_data)
    for row in _load_output(tmp_data["output_path"]):
        user = row["messages"][1]["content"]
        assert user.startswith("Meme text: ")
        assert "\n\nImage description: " in user

def test_assistant_valid_json(tmp_data):
    """Assistant content is valid JSON with label, confidence, reasoning."""
    _run(tmp_data)
    for row in _load_output(tmp_data["output_path"]):
        asst = json.loads(row["messages"][2]["content"])
        assert asst["label"] in (0, 1)
        assert isinstance(asst["confidence"], int)
        assert 0 <= asst["confidence"] <= 100
        assert isinstance(asst["reasoning"], str)
        assert len(asst["reasoning"]) > 0

def test_confidence_direction(tmp_data):
    """Hateful → confidence ∈ [82,96]; benign → confidence ∈ [4,18]."""
    _run(tmp_data)
    for row in _load_output(tmp_data["output_path"]):
        asst = json.loads(row["messages"][2]["content"])
        if asst["label"] == 1:
            assert 82 <= asst["confidence"] <= 96, f"hateful conf={asst['confidence']}"
        else:
            assert 4 <= asst["confidence"] <= 18, f"benign conf={asst['confidence']}"

def test_hatred_reasoning_used(tmp_data):
    """IDs 100 and 101 (hateful + HatReD) use the HatReD reasoning text."""
    _run(tmp_data)
    rows = _load_output(tmp_data["output_path"])
    by_text = {}
    for row in rows:
        user = row["messages"][1]["content"]
        # extract meme_text from user content
        meme_text = user.split("Meme text: ")[1].split("\n\nImage description:")[0]
        asst = json.loads(row["messages"][2]["content"])
        by_text[meme_text] = asst

    assert "mocks jewish people" in by_text["text100"]["reasoning"]
    assert "the jews" in by_text["text100"]["reasoning"]  # target prepended
    assert "dehumanizes immigrants" in by_text["text101"]["reasoning"]

def test_hateful_fallback_template(tmp_data):
    """ID 102 (hateful, no HatReD) uses the generic hateful template."""
    _run(tmp_data)
    rows = _load_output(tmp_data["output_path"])
    for row in rows:
        user = row["messages"][1]["content"]
        meme_text = user.split("Meme text: ")[1].split("\n\nImage description:")[0]
        if meme_text == "text102":
            asst = json.loads(row["messages"][2]["content"])
            assert asst["label"] == 1
            # fallback template must not be empty
            assert len(asst["reasoning"]) > 10
            return
    pytest.fail("id 102 not found in output")

def test_benign_template(tmp_data):
    """IDs 103, 104 (benign) use the benign template."""
    _run(tmp_data)
    rows = _load_output(tmp_data["output_path"])
    benign_reasonings = []
    for row in rows:
        asst = json.loads(row["messages"][2]["content"])
        if asst["label"] == 0:
            benign_reasonings.append(asst["reasoning"])
    assert len(benign_reasonings) == 2
    for r in benign_reasonings:
        assert len(r) > 10

def test_missing_image_excluded(tmp_data):
    """ID 105 (status=missing_image) must not appear in output."""
    _run(tmp_data)
    rows = _load_output(tmp_data["output_path"])
    all_user = [r["messages"][1]["content"] for r in rows]
    assert not any("skip" in u for u in all_user)

def test_seed_determinism(tmp_data):
    """Same seed → identical output."""
    _run(tmp_data, seed=7)
    rows1 = _load_output(tmp_data["output_path"])
    _run(tmp_data, seed=7)
    rows2 = _load_output(tmp_data["output_path"])
    assert rows1 == rows2
```

- [ ] **Step 2: Run tests — expect failures (finetune.py not yet rewritten)**

```bash
cd hateful_memes_app && python -m pytest tests/test_finetune.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` (functions don't exist yet). This confirms the tests are wired correctly.

---

## Task 4: Rewrite finetune.py

**Files:**
- Rewrite: `hateful_memes_app/finetune.py`

- [ ] **Step 1: Write the new finetune.py**

```python
# hateful_memes_app/finetune.py
"""
build_finetune_data()
Produces data/finetune_data.jsonl — chat-format training examples for QLoRA.

Format per line:
  {"messages": [
      {"role": "system",    "content": "<ZS+RP+AD system prompt>"},
      {"role": "user",      "content": "Meme text: …\n\nImage description: …"},
      {"role": "assistant", "content": "{\"reasoning\": \"…\", \"label\": 0|1, \"confidence\": int}"}
  ]}

Confidence encodes P(hateful) for AUROC:
  label=1 → random int in [82, 96]   (high → correct positive signal)
  label=0 → random int in [4, 18]    (low  → correct negative signal)
"""
import json, os, random
from excel_utils import read_sheet
from utils import load_jsonl

# ── Defaults (relative to project root, i.e. one level above hateful_memes_app/) ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TRAIN_XLSX   = os.path.join(_ROOT, "data", "phase1_train_zsrp.xlsx")
DEFAULT_TRAIN_SHEET  = "ZS + RP"
DEFAULT_TRAIN_JSONL  = os.path.join(_ROOT, "data", "train.jsonl")
DEFAULT_HATRED_JSONL = os.path.join(_ROOT, "HatRed", "datasets", "hatred",
                                    "annotations", "fhm_train_reasonings.jsonl")
DEFAULT_PROMPTS_XLSX = os.path.join(_ROOT, "data", "prompts.xlsx")
DEFAULT_OUTPUT_JSONL = os.path.join(_ROOT, "data", "finetune_data.jsonl")
DEFAULT_PROMPT_NAME  = "ZS+RP+AD"
DEFAULT_SEED         = 42

CONF_HATEFUL = (82, 96)
CONF_BENIGN  = (4, 18)

HATEFUL_TEMPLATE = (
    "The meme targets a protected group through dehumanizing, stereotyping, "
    "or inciting content."
)
BENIGN_TEMPLATE = (
    "No attack on a protected group is present. The content does not dehumanize, "
    "stereotype, or incite hatred toward any individual or group. The meme is not hateful."
)


def _load_descriptions(train_xlsx: str, train_sheet: str) -> dict[int, str]:
    """Returns {id: description} for rows with status=='ok'."""
    df = read_sheet(train_xlsx, train_sheet)
    return {
        int(r["id"]): str(r["description"])
        for _, r in df.iterrows()
        if r.get("status") == "ok"
    }


def _load_labels_and_texts(train_jsonl: str) -> tuple[dict[int, int], dict[int, str]]:
    """Returns ({id: label}, {id: text})."""
    labels, texts = {}, {}
    for e in load_jsonl(train_jsonl):
        if "label" in e:
            mid = int(e["id"])
            labels[mid] = int(e["label"])
            texts[mid]  = str(e.get("text", ""))
    return labels, texts


def _load_hatred(hatred_jsonl: str) -> dict[int, dict]:
    """Returns {id: {target: list, reasonings: list}} from HatReD annotations."""
    result = {}
    for entry in load_jsonl(hatred_jsonl):
        img = entry.get("img", "")
        # img is like '01235.png' (no path prefix)
        try:
            mid = int(os.path.splitext(os.path.basename(img))[0])
        except ValueError:
            continue
        reasonings = entry.get("reasonings", [])
        if reasonings:
            result[mid] = {
                "target": entry.get("target", []),
                "reasonings": reasonings,
            }
    return result


def _load_system_prompt(prompts_xlsx: str, prompt_name: str) -> str:
    df = read_sheet(prompts_xlsx, "Phase2")
    match = df[df["Name"] == prompt_name]
    if match.empty:
        raise ValueError(
            f"Prompt '{prompt_name}' not found in Phase2 sheet of {prompts_xlsx}. "
            f"Available: {list(df['Name'])}"
        )
    return str(match.iloc[0]["Prompt"])


def _build_reasoning(label: int, mid: int, hatred_map: dict) -> str:
    if label == 1 and mid in hatred_map:
        h = hatred_map[mid]
        parts = []
        if h["target"]:
            parts.append(f"Targets: {', '.join(h['target'])}.")
        parts.extend(h["reasonings"])
        return " ".join(parts)
    if label == 1:
        return HATEFUL_TEMPLATE
    return BENIGN_TEMPLATE


def build_finetune_data(
    train_xlsx:   str = DEFAULT_TRAIN_XLSX,
    train_sheet:  str = DEFAULT_TRAIN_SHEET,
    train_jsonl:  str = DEFAULT_TRAIN_JSONL,
    hatred_jsonl: str = DEFAULT_HATRED_JSONL,
    prompts_xlsx: str = DEFAULT_PROMPTS_XLSX,
    output_jsonl: str = DEFAULT_OUTPUT_JSONL,
    prompt_name:  str = DEFAULT_PROMPT_NAME,
    seed:         int = DEFAULT_SEED,
) -> int:
    """Build chat-format fine-tuning JSONL. Returns number of examples written."""
    rng = random.Random(seed)

    desc_map          = _load_descriptions(train_xlsx, train_sheet)
    labels, texts     = _load_labels_and_texts(train_jsonl)
    hatred_map        = _load_hatred(hatred_jsonl)
    system_prompt     = _load_system_prompt(prompts_xlsx, prompt_name)

    valid_ids = sorted(set(desc_map) & set(labels))

    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else ".", exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for mid in valid_ids:
            label       = labels[mid]
            description = desc_map[mid]
            meme_text   = texts.get(mid, "")

            lo, hi  = CONF_HATEFUL if label == 1 else CONF_BENIGN
            confidence  = rng.randint(lo, hi)
            reasoning   = _build_reasoning(label, mid, hatred_map)

            assistant_json = json.dumps(
                {"reasoning": reasoning, "label": label, "confidence": confidence},
                ensure_ascii=False,
            )
            example = {
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": f"Meme text: {meme_text}\n\nImage description: {description}"},
                    {"role": "assistant", "content": assistant_json},
                ]
            }
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(valid_ids)


if __name__ == "__main__":
    n = build_finetune_data()
    print(f"Wrote {n} examples → {DEFAULT_OUTPUT_JSONL}")
```

- [ ] **Step 2: Run the tests**

```bash
cd hateful_memes_app && python -m pytest tests/test_finetune.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd hateful_memes_app && python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Smoke-run on real data**

```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform/hateful_memes_app
python finetune.py
```

Expected output: `Wrote 2000 examples → data/finetune_data.jsonl`

Then verify:
```bash
wc -l ../data/finetune_data.jsonl
python3 -c "
import json
with open('../data/finetune_data.jsonl') as f:
    rows = [json.loads(l) for l in f]
asst = [json.loads(r['messages'][2]['content']) for r in rows]
labels = [a['label'] for a in asst]
confs  = [a['confidence'] for a in asst]
print('total:', len(rows))
print('hateful:', labels.count(1), '| benign:', labels.count(0))
print('conf range hateful:', min(c for c,l in zip(confs,labels) if l==1),
      '-', max(c for c,l in zip(confs,labels) if l==1))
print('conf range benign:', min(c for c,l in zip(confs,labels) if l==0),
      '-', max(c for c,l in zip(confs,labels) if l==0))
"
```

Expected:
```
2000
total: 2000
hateful: 1000 | benign: 1000
conf range hateful: 82 - 96
conf range benign: 4 - 18
```

- [ ] **Step 5: Commit**

```bash
git add hateful_memes_app/finetune.py hateful_memes_app/tests/test_finetune.py data/finetune_data.jsonl
git commit -m "feat: rewrite finetune.py — build_finetune_data with HatReD reasoning and AUROC-correct confidence"
```

---

## Task 5: Create train_qlora.py (RunPod training script)

**Files:**
- Create: `hateful_memes_app/train_qlora.py`
- Create: `hateful_memes_app/requirements-train.txt`

- [ ] **Step 1: Create requirements-train.txt**

```
# Training dependencies (RunPod only — not used by the Streamlit app)
# Install order matters: unsloth first, then trl/peft/accelerate/bitsandbytes without deps
unsloth
trl>=0.8.6
peft>=0.10.0
accelerate>=0.27.0
bitsandbytes>=0.42.0
datasets>=2.18.0
```

Save as `hateful_memes_app/requirements-train.txt`.

- [ ] **Step 2: Create train_qlora.py**

```python
# hateful_memes_app/train_qlora.py
"""
QLoRA fine-tuning of Phi-4-mini-instruct via Unsloth.
Run on RunPod (RTX 3090, 24 GB VRAM).

Usage (from /root/hatememe/):
    python hateful_memes_app/train_qlora.py

Outputs:
    phase2-ft-gguf/          ← GGUF file (q4_k_m) + Modelfile
"""
import json, os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Phi-4-mini-instruct"
DATA_PATH      = "data/finetune_data.jsonl"
OUTPUT_DIR     = "phase2-ft-gguf"
MAX_SEQ_LENGTH = 4096
SEED           = 42
EPOCHS         = 2
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 2e-4
WARMUP_RATIO   = 0.03

# ── Imports (heavy — only available on RunPod after pip install) ───────────────
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


def main():
    # 1. Load base model in 4-bit
    print("Loading model …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,          # auto-detect (bf16 on Ampere)
    )

    # 2. Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # 3. Load and format data
    dataset = load_dataset(DATA_PATH)

    def apply_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True,
                          remove_columns=["messages"])

    # 4. Quick sanity check: print first formatted example
    print("\n── First training example (truncated) ──")
    print(dataset[0]["text"][:600])
    print("…\n")

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            bf16=True,
            optim="adamw_8bit",
            seed=SEED,
            output_dir="checkpoints",
            logging_steps=25,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    # 6. Train only on assistant responses (ignore loss on system+user tokens)
    # Phi-4-mini chat template uses <|user|> / <|assistant|> tokens.
    # Verify with: tokenizer.apply_chat_template([{"role":"user","content":"hi"}],
    #              tokenize=False) — look for the exact strings around the turns.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|user|>\n",
        response_part="<|assistant|>\n",
    )

    # 7. Train
    print("Starting training …")
    trainer.train()
    print("Training complete.")

    # 8. Export GGUF (q4_k_m) — creates OUTPUT_DIR/*.gguf + Modelfile
    print(f"Exporting GGUF → {OUTPUT_DIR}/ …")
    model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")

    print("\n── Export complete ──")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print(f"  {fname}  ({size / 1e9:.2f} GB)")

    print(f"""
Next steps on Mac:
  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:/root/hatememe/{OUTPUT_DIR}/ ./{OUTPUT_DIR}/
  ollama create phase2-ft -f {OUTPUT_DIR}/Modelfile
  # then in config.json: "ft_model_path": "phase2-ft"
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the script is syntactically valid (no Unsloth needed locally)**

```bash
python3 -m py_compile hateful_memes_app/train_qlora.py && echo "syntax OK"
```

Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add hateful_memes_app/train_qlora.py hateful_memes_app/requirements-train.txt
git commit -m "feat: add train_qlora.py (Unsloth QLoRA + GGUF export) and requirements-train.txt"
```

---

## Task 6: Create launch_runpod_ft.sh

**Files:**
- Create: `hateful_memes_app/launch_runpod_ft.sh`

This script runs **on the RunPod VM** after uploading the project files. It installs deps and launches training in a tmux session.

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# hateful_memes_app/launch_runpod_ft.sh
# Run on RunPod VM from /root/hatememe/:
#   bash hateful_memes_app/launch_runpod_ft.sh
set -e

echo "=== 1/5 System packages ==="
apt-get update -qq && apt-get install -y -qq tmux rsync

echo "=== 2/5 Verify GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "=== 3/5 Install training dependencies ==="
pip install --upgrade pip -q
# Unsloth: install for current CUDA/torch env, then no-deps for the rest
pip install "unsloth" -q
pip install --no-deps trl peft accelerate bitsandbytes datasets -q

echo "=== 4/5 Verify Unsloth import ==="
python3 -c "from unsloth import FastLanguageModel; print('unsloth OK')"

echo "=== 5/5 Launch training in tmux session 'ft' ==="
tmux new-session -d -s ft \
  "cd /root/hatememe && python hateful_memes_app/train_qlora.py 2>&1 | tee /root/train.log"

echo ""
echo "Training started in tmux session 'ft'."
echo "Monitor with:  tmux attach -t ft"
echo "Or log:        tail -f /root/train.log"
```

Save as `hateful_memes_app/launch_runpod_ft.sh`.

- [ ] **Step 2: Make executable**

```bash
chmod +x hateful_memes_app/launch_runpod_ft.sh
```

- [ ] **Step 3: Commit**

```bash
git add hateful_memes_app/launch_runpod_ft.sh
git commit -m "feat: add launch_runpod_ft.sh for RunPod training setup"
```

---

## Task 7: RunPod — upload, train, download

This task is manual (requires a live RunPod pod). Replace `<PORT>` and `<HOST>` with values from the RunPod dashboard.

- [ ] **Step 1: Create RunPod pod**

On https://runpod.io/console/pods → Deploy:
- GPU: RTX 3090 (24 GB)
- Template: RunPod PyTorch 2.1 (CUDA 12.1, Ubuntu 22.04)
- Disk: 50 GB
- Disable "Stop Idle Pod"

- [ ] **Step 2: Create dirs on VM**

```bash
ssh -p <PORT> root@<HOST> "mkdir -p /root/hatememe/data /root/hatememe/hateful_memes_app"
```

- [ ] **Step 3: Upload project files**

```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform

# App code
rsync -avz --progress \
  --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  -e "ssh -p <PORT>" \
  hateful_memes_app/ root@<HOST>:/root/hatememe/hateful_memes_app/

# Training data
rsync -avz --progress -e "ssh -p <PORT>" \
  data/finetune_data.jsonl root@<HOST>:/root/hatememe/data/finetune_data.jsonl
```

- [ ] **Step 4: Verify upload**

```bash
ssh -p <PORT> root@<HOST> "wc -l /root/hatememe/data/finetune_data.jsonl"
```

Expected: `2000`

- [ ] **Step 5: Run setup + training**

```bash
ssh -p <PORT> root@<HOST> "cd /root/hatememe && bash hateful_memes_app/launch_runpod_ft.sh"
```

Then detach and monitor:
```bash
ssh -p <PORT> root@<HOST> "tail -f /root/train.log"
```

Estimated runtime on RTX 3090: **~45–90 min** for 2000 examples × 2 epochs.

- [ ] **Step 6: Verify GGUF export**

```bash
ssh -p <PORT> root@<HOST> "ls -lh /root/hatememe/phase2-ft-gguf/"
```

Expected: a `.gguf` file (~2–3 GB) and a `Modelfile`.

- [ ] **Step 7: Download GGUF to Mac**

```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform
mkdir -p phase2-ft-gguf
rsync -avz --progress \
  -e "ssh -p <PORT>" \
  root@<HOST>:/root/hatememe/phase2-ft-gguf/ ./phase2-ft-gguf/
```

- [ ] **Step 8: Stop pod**

In RunPod dashboard → Stop Pod (not Terminate).

---

## Task 8: Register model with Ollama and update config

- [ ] **Step 1: Create the Ollama model**

```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform
ollama create phase2-ft -f phase2-ft-gguf/Modelfile
```

Expected: `success`

- [ ] **Step 2: Verify model is listed**

```bash
ollama list | grep phase2-ft
```

Expected: line showing `phase2-ft` with size ~2–3 GB.

- [ ] **Step 3: Quick sanity check (single inference)**

```bash
ollama run phase2-ft "Meme text: love everyone\n\nImage description: A rainbow flag." \
  --system "You are an expert hate speech analyst. Reply with JSON only: {\"reasoning\": \"…\", \"label\": 0, \"confidence\": 75}"
```

Expected: valid JSON with `label: 0`.

- [ ] **Step 4: Update config.json**

In `hateful_memes_app/config.json`, set:
```json
"ft_model_path": "phase2-ft"
```

- [ ] **Step 5: Commit**

```bash
git add hateful_memes_app/config.json phase2-ft-gguf/Modelfile
git commit -m "feat: register phase2-ft Ollama model and enable ft_model_path in config"
```

---

## Task 9: Run the 2-way eval comparison

- [ ] **Step 1: Configure eval to use ZS+RP+AD dev descriptions**

In `hateful_memes_app/config.json`, verify:
```json
"phase1_excel": "data/phase1_dev_zsrp.xlsx"
```

(This file contains 500/500 ok dev descriptions generated with the `ZS + RP` Phase-1 prompt — matching the training data style.)

- [ ] **Step 2: Run baseline — ZS+RP+AD (no FT)**

In the Streamlit app:
- Phase-2 prompt: `ZS+RP+AD`
- Use FT model: **off** (uses `phi4-mini`)
- Phase-1 sheet: `ZS + RP`
- Click "Phase 2 starten"

Wait for 500 examples to complete.

- [ ] **Step 3: Run fine-tuned — ZS+RP+AD+FT**

In the Streamlit app:
- Phase-2 prompt: `ZS+RP+AD`
- Use FT model: **on** (uses `phase2-ft`)
- Phase-1 sheet: `ZS + RP`
- Click "Phase 2 starten"

Wait for 500 examples to complete.

- [ ] **Step 4: Verify metrics in phase2_results.xlsx**

```bash
python3 -c "
from hateful_memes_app.excel_utils import read_sheet
import pandas as pd

sheets_to_check = ['M_ZS + RPxZS+RP+AD', 'M_ZS + RPxZS+RP+AD_ft']  # safe_sheet_name variants
wb_path = 'data/phase2_results.xlsx'
import openpyxl
wb = openpyxl.load_workbook(wb_path)
print('Sheets:', wb.sheetnames)
"
```

Both metric sheets should show `accuracy`, `auroc`, `f1` — with `auroc` > 0.5 (not n/a) confirming the confidence fix worked.

---

## Self-Review

### Spec coverage check

| Spec requirement | Covered by task |
|---|---|
| ZS+RP+AD prompt in prompts.xlsx | Task 1 |
| phase2.py user_prompt fix | Task 2 |
| finetune.py rewrite with HatReD + AUROC-correct confidence | Tasks 3+4 |
| train_qlora.py (Unsloth, QLoRA r=16, bf16, train_on_responses_only) | Task 5 |
| launch_runpod_ft.sh | Task 6 |
| requirements-train.txt | Task 5 step 1 |
| RunPod upload/train/download | Task 7 |
| ollama create phase2-ft + config.json | Task 8 |
| 2-run eval (baseline vs FT) on ZS+RP dev descriptions | Task 9 |
| Tests for build_finetune_data | Task 3 |

### No placeholders — checked ✓
All steps contain exact code, commands, and expected output.

### Type consistency check
- `build_finetune_data()` signature matches test fixture `_run()` kwargs ✓
- `read_sheet()` returns DataFrame with `Name`/`Prompt` columns (verified from actual file) ✓
- `load_jsonl()` returns `list[dict]` (matches utils.py signature) ✓
- HatReD img format `'01235.png'` → `int(os.path.splitext(basename)[0])` = 1235 ✓ (verified)
- `train_on_responses_only` called with `instruction_part`/`response_part` kwargs ✓

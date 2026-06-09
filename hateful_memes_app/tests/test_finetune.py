# hateful_memes_app/tests/test_finetune.py
import json, os, tempfile
import openpyxl
import pytest
from finetune import build_finetune_data

# ── Fixtures ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a test analyst. Reply with JSON only."


def _make_phase1_xlsx(path: str, rows: list) -> None:
    """rows: list of {id, text, description, status}"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ZS + RP"
    ws.append(["id", "img", "text", "description", "prompt_name", "status"])
    for r in rows:
        ws.append([r["id"], f"img/{r['id']:05d}.png", r.get("text", ""),
                   r["description"], "ZS + RP", r.get("status", "ok")])
    wb.save(path)


def _make_prompts_xlsx(path: str, prompt_name: str, prompt_text: str) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Phase2"
    ws.append(["Name", "Prompt"])
    ws.append([prompt_name, prompt_text])
    wb.save(path)


def _make_train_jsonl(path: str, rows: list) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_hatred_jsonl(path: str, rows: list) -> None:
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
    6 memes: ids 100–105
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
        {"id": 101, "target": [None],       "reasonings": ["dehumanizes immigrants."]},
    ]

    phase1_path  = str(tmp_path / "phase1.xlsx")
    prompts_path = str(tmp_path / "prompts.xlsx")
    train_path   = str(tmp_path / "train.jsonl")
    hatred_path  = str(tmp_path / "hatred.jsonl")
    output_path  = str(tmp_path / "finetune_data.jsonl")

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


# ── Tests ─────────────────────────────────────────────────────────────────────

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

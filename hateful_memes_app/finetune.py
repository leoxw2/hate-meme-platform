# hateful_memes_app/finetune.py
"""
build_finetune_data()
Produces data/finetune_data.jsonl — chat-format training examples for QLoRA.

Format per line:
  {"messages": [
      {"role": "system",    "content": "<ZS+RP+AD system prompt>"},
      {"role": "user",      "content": "Meme text: …\\n\\nImage description: …"},
      {"role": "assistant", "content": '{"reasoning": "…", "label": 0|1, "confidence": int}'}
  ]}

Confidence encodes P(hateful) for AUROC:
  label=1 → random int in [82, 96]   (high → correct positive signal)
  label=0 → random int in [4, 18]    (low  → correct negative signal)
"""
import json, os, random
from excel_utils import read_sheet
from utils import load_jsonl

# ── Defaults (relative to project root, one level above hateful_memes_app/) ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TRAIN_XLSX   = os.path.join(_ROOT, "data", "phase1_train_zsrp.xlsx")
DEFAULT_TRAIN_SHEET  = "ZS + RP"
DEFAULT_TRAIN_JSONL  = os.path.join(_ROOT, "data", "train.jsonl")
DEFAULT_HATRED_JSONL = os.path.join(_ROOT, "HatRed", "datasets", "hatred",
                                    "annotations", "fhm_train_reasonings.jsonl")
DEFAULT_RACE_JSONL   = os.path.join(_ROOT, "HatRed", "datasets", "hatred",
                                    "auxiliary", "fhm_train_race.jsonl")
DEFAULT_PROMPTS_XLSX = os.path.join(_ROOT, "data", "prompts.xlsx")
DEFAULT_OUTPUT_JSONL = os.path.join(_ROOT, "data", "finetune_data.jsonl")
DEFAULT_PROMPT_NAME  = "ZS+RP+AD"
DEFAULT_SEED         = 42

CONF_HATEFUL = (82, 96)
CONF_BENIGN  = (4, 18)

# Hateful memes WITHOUT real HatReD reasoning are excluded from training
# (we only train hateful examples that carry a genuine human-written rationale).
BENIGN_TEMPLATE = (
    "No attack on a protected group is present. The content does not dehumanize, "
    "stereotype, or incite hatred toward any individual or group. The meme is not hateful."
)


def _load_descriptions(train_xlsx: str, train_sheet: str) -> dict:
    """Returns {id: description} for rows with status=='ok'."""
    df = read_sheet(train_xlsx, train_sheet)
    return {
        int(r["id"]): str(r["description"])
        for _, r in df.iterrows()
        if r.get("status") == "ok"
    }


def _load_labels_and_texts(train_jsonl: str) -> tuple:
    """Returns ({id: label}, {id: text})."""
    labels, texts = {}, {}
    for e in load_jsonl(train_jsonl):
        if "label" in e:
            mid = int(e["id"])
            labels[mid] = int(e["label"])
            texts[mid]  = str(e.get("text", ""))
    return labels, texts


def _load_hatred(hatred_jsonl: str) -> dict:
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


def _load_race(race_jsonl: str) -> dict:
    """Returns {id: race_string} from HatReD auxiliary race annotations.

    Returns an empty dict if the file is missing (race enrichment is optional).
    """
    if not os.path.exists(race_jsonl):
        return {}
    result = {}
    for entry in load_jsonl(race_jsonl):
        img = entry.get("img", "")
        try:
            mid = int(os.path.splitext(os.path.basename(img))[0])
        except ValueError:
            continue
        race = entry.get("race")
        if race:
            result[mid] = str(race)
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


def _build_reasoning(label: int, mid: int, hatred_map: dict, race_map: dict) -> str:
    """Build the assistant reasoning text.

    For hateful memes (label==1) the caller guarantees mid is in hatred_map.
    When a race annotation exists for the meme, it enriches the target prefix
    (e.g. "Targets: the muslim (Middle Eastern Male).").
    """
    if label == 1:
        h = hatred_map[mid]
        targets = [t for t in h["target"] if t is not None]
        parts = []
        if targets:
            target_str = ", ".join(targets)
            race = race_map.get(mid)
            if race:
                parts.append(f"Targets: {target_str} ({race}).")
            else:
                parts.append(f"Targets: {target_str}.")
        parts.extend(h["reasonings"])
        return " ".join(parts)
    return BENIGN_TEMPLATE


def build_finetune_data(
    train_xlsx:   str = DEFAULT_TRAIN_XLSX,
    train_sheet:  str = DEFAULT_TRAIN_SHEET,
    train_jsonl:  str = DEFAULT_TRAIN_JSONL,
    hatred_jsonl: str = DEFAULT_HATRED_JSONL,
    race_jsonl:   str = DEFAULT_RACE_JSONL,
    prompts_xlsx: str = DEFAULT_PROMPTS_XLSX,
    output_jsonl: str = DEFAULT_OUTPUT_JSONL,
    prompt_name:  str = DEFAULT_PROMPT_NAME,
    seed:         int = DEFAULT_SEED,
) -> int:
    """Build chat-format fine-tuning JSONL. Returns number of examples written.

    Hateful memes (label==1) WITHOUT a real HatReD reasoning are skipped — we
    only train hateful examples that carry a genuine human-written rationale.
    All benign memes (label==0) are kept and use the benign template.
    """
    rng = random.Random(seed)

    desc_map      = _load_descriptions(train_xlsx, train_sheet)
    labels, texts = _load_labels_and_texts(train_jsonl)
    hatred_map    = _load_hatred(hatred_jsonl)
    race_map      = _load_race(race_jsonl)
    system_prompt = _load_system_prompt(prompts_xlsx, prompt_name)

    valid_ids = sorted(set(desc_map) & set(labels))

    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for mid in valid_ids:
            label = labels[mid]

            # Skip hateful memes that have no human-written HatReD reasoning.
            if label == 1 and mid not in hatred_map:
                continue

            description = desc_map[mid]
            meme_text   = texts.get(mid, "")

            lo, hi     = CONF_HATEFUL if label == 1 else CONF_BENIGN
            confidence = rng.randint(lo, hi)
            reasoning  = _build_reasoning(label, mid, hatred_map, race_map)

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
            written += 1

    return written


if __name__ == "__main__":
    n = build_finetune_data()
    print(f"Wrote {n} examples → {DEFAULT_OUTPUT_JSONL}")

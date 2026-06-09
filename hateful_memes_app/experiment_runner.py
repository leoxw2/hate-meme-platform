import os
import pandas as pd
from phase1 import run_phase1
from phase2 import run_phase2
from excel_utils import safe_sheet_name, append_to_csv, get_sheet_names

# (combo_name) -> (base_phase2_prompt, use_rag, use_ft)
#
# FT isolation: the fine-tuned model was trained with the ZS+RP+AD system prompt.
# To attribute a metric delta to fine-tuning (and not to a prompt switch), the FT
# cell MUST evaluate under the SAME prompt it was trained on, paired against the
# non-FT baseline under that identical prompt. Hence the matched pair
# ZS+RP+AD (baseline) vs ZS+RP+AD+FT (fine-tuned). The earlier ZS+FT / CoT+FS+AD+FT
# cells were removed: they ran the FT model under prompts it never saw in training,
# confounding the FT effect with a prompt change.
PHASE2_CONFIG = {
    "ZS":            ("ZS",        False, False),
    "CoT+FS+AD":     ("CoT+FS+AD", False, False),
    "CoT+FS+AD+RAG": ("CoT+FS+AD", True,  False),
    "ZS+RAG":        ("ZS",        True,  False),
    "ZS+RP+AD":      ("ZS+RP+AD",  False, False),  # FT baseline (matched prompt)
    "ZS+RP+AD+FT":   ("ZS+RP+AD",  False, True),   # fine-tuned (matched prompt)
}

def _runner_csv(results_folder: str) -> str:
    return os.path.join(results_folder, "runner_completed.csv")

def run_experiments(phase1_selections: list[str], phase2_selections: list[str],
                    prompts_phase1: dict[str, str], prompts_phase2: dict[str, str],
                    phase1_excel: str, dev_jsonl_path: str,
                    phase2_excel: str, results_folder: str,
                    img_folder: str = "",
                    max_tokens: int = 3000, max_time_secs: int = 150,
                    ft_model_path: str = "", rag_retriever=None):
    """Generator: läuft Phase 1 (alle ausgewählten Prompts) und danach
    alle Phase1 × Phase2 Kombinationen durch.
    Checkpoint: CSV mit abgeschlossenen Keys.
    """
    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    # Bereits vorhandene Phase-1-Sheets ermitteln (Resume)
    existing_p1_sheets = set()
    if os.path.exists(phase1_excel):
        try:
            existing_p1_sheets = set(get_sheet_names(phase1_excel))
        except Exception:
            pass

    for i, p1_name in enumerate(phase1_selections):
        sheet_name = safe_sheet_name(p1_name)
        if sheet_name in existing_p1_sheets:
            yield {"type": "phase1_skip", "prompt": p1_name,
                   "index": i + 1, "total": len(phase1_selections)}
            continue

        prompt_text = prompts_phase1.get(p1_name, "")
        yield {"type": "phase1_start", "prompt": p1_name,
               "index": i + 1, "total": len(phase1_selections)}

        for update in run_phase1(
            jsonl_path=dev_jsonl_path,
            img_folder=img_folder,
            phase1_excel=phase1_excel,
            prompt_name=p1_name,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            max_time_secs=max_time_secs,
            results_folder=results_folder,
        ):
            yield update

        yield {"type": "phase1_done", "prompt": p1_name}

    # ── PHASE 2 ───────────────────────────────────────────────────────────────
    if not phase2_selections:
        yield {"type": "all_done", "completed": 0}
        return

    combinations = [(p1, p2)
                    for p1 in phase1_selections
                    for p2 in phase2_selections]
    total = len(combinations)

    runner_csv = _runner_csv(results_folder)
    completed = set()
    if os.path.exists(runner_csv):
        completed = set(pd.read_csv(runner_csv)["combo_key"].astype(str))

    n_done = 0
    for i, (phase1_sheet, phase2_name) in enumerate(combinations):
        combo_key = safe_sheet_name(f"{phase1_sheet}x{phase2_name}")
        if combo_key in completed:
            yield {"type": "combination_skip", "phase1": phase1_sheet,
                   "phase2": phase2_name, "index": i + 1, "total": total}
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

        append_to_csv({"combo_key": combo_key}, runner_csv)
        completed.add(combo_key)
        n_done += 1

        yield {"type": "combination_done", "phase1": phase1_sheet,
               "phase2": phase2_name, "metrics": metrics}

    yield {"type": "all_done", "completed": n_done}

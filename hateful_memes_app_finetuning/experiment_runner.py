import os
import pandas as pd
from phase2 import run_phase2
from excel_utils import safe_sheet_name, append_to_csv

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

    runner_csv = _runner_csv(results_folder)
    completed = set()
    if os.path.exists(runner_csv):
        completed = set(pd.read_csv(runner_csv)["combo_key"].astype(str))

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

        yield {"type": "combination_done", "phase1": phase1_sheet,
               "phase2": phase2_name, "metrics": metrics}

    yield {"type": "all_done", "completed": len(completed)}

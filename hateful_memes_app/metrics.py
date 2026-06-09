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
        raw_auroc = roc_auc_score(y_true, y_prob_norm)
        import math
        auroc = round(raw_auroc, 4) if not math.isnan(raw_auroc) else "n/a"
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

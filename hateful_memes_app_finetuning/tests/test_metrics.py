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

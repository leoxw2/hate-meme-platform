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

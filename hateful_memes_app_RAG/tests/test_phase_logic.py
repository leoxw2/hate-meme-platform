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

        processed_ids = set(pd.read_csv(csv_path)["id"].astype(int))
        assert 100 in processed_ids
        assert 200 in processed_ids
        assert len(processed_ids) == 2

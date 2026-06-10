import json, tempfile, os
from utils import load_jsonl

def test_load_jsonl_basic():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.jsonl")
        with open(path, "w") as f:
            f.write('{"id": 1, "text": "hello"}\n')
            f.write('\n')
            f.write('{"id": 2, "text": "world"}\n')
        result = load_jsonl(path)
        assert len(result) == 2
        assert result[0]["id"] == 1

def test_load_jsonl_empty_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty.jsonl")
        open(path, "w").close()
        assert load_jsonl(path) == []

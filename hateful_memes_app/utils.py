import json

def load_jsonl(path: str) -> list[dict]:
    """Liest eine JSONL-Datei ein. Leere Zeilen werden übersprungen."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

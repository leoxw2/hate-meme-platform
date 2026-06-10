# Project Overview — Hateful Memes Classification Platform

> Dieses Dokument dient als Kontext-Briefing für neue Claude-Sessions.  
> Bitte am Anfang jeder Session teilen: "Lies bitte projectOverview.md"

---

## 1. Was ist das Projekt?

Wissenschaftliches Forschungsprojekt (Bachelorarbeit / PFT) zur automatischen Erkennung von Hate Speech in Internet-Memes.

**Kernfrage:** Welche Kombination aus Phase-1-Prompt × Phase-2-Prompt erzielt die beste Klassifizierungsleistung bei Hateful Memes — gemessen an Accuracy, AUROC und F1?

**Hypothese:** Eine detailliertere Bildbeschreibung in Phase 1 (z.B. mit Chain-of-Thought) kombiniert mit einem strukturierten Phase-2-Prompt (z.B. CoT+Few-Shot) liefert bessere Metriken als einfache Zero-Shot-Ansätze.

**Ansatz:** Statt ein einziges multimodales Modell zu nutzen, wird die Aufgabe in zwei Stufen zerlegt:
1. **Phase 1** — Ein multimodales Modell beschreibt das Bild (Text-Output)
2. **Phase 2** — Ein reines Sprachmodell klassifiziert anhand der Beschreibung + Meme-Text

**Ziel:** Die beste Phase-1 × Phase-2-Kombination identifizieren und die Auswirkung jeder Strategie (RAG, Fine-Tuning, CoT) isoliert bewerten.

---

## 2. Architektur

```
Meme-Bild (PNG)
      │
      ▼
┌─────────────────────────────────────┐
│  Phase 1: QWEN2.5-VL-3B (multimodal)│
│  Prompt → strukturierte Beschreibung│
│  "Was zeigt das Bild? Welche        │
│   Gruppen werden erwähnt? ..."      │
└──────────────────┬──────────────────┘
                   │ Textbeschreibung
                   ▼
┌─────────────────────────────────────┐
│  Phase 2: PHI-4-MINI (LLM)         │
│  Input: Beschreibung + Meme-Text   │
│  Output: JSON                      │
│  {"label": 0,                      │
│   "confidence": 75,                │
│   "reasoning": "..."}              │
└──────────────────┬──────────────────┘
                   │
                   ▼
         Excel-Ergebnisse
         Metriken: Accuracy, AUROC, F1
```

Beide Modelle laufen **lokal via Ollama** (kein API-Key, keine Kosten, kein Internet nötig).

---

## 3. Tech Stack

| Komponente | Details |
|------------|---------|
| **Frontend** | Streamlit (Python), 4 Tabs |
| **Phase-1-Modell** | `qwen2.5vl:3b` via Ollama (multimodal, 2.2 GB VRAM) |
| **Phase-2-Modell** | `phi4-mini:latest` via Ollama (LLM) |
| **Inferenz** | Ollama REST API (localhost:11434), 100% GPU |
| **Datenhaltung** | Excel (openpyxl) + CSV als Lauf-Checkpoint |
| **RAG** | ChromaDB + sentence-transformers |
| **Metriken** | scikit-learn: Accuracy, AUROC, F1 |
| **Python** | 3.12 |
| **Betriebssystem** | Windows 11 |

---

## 4. Datensatz

**Hateful Memes Dataset** (Meta AI Research, 2020)  
→ Paper: "The Hateful Memes Challenge" (Kiela et al.)

| Eigenschaft | Wert |
|-------------|------|
| Datei | `data/dev.jsonl` |
| Einträge | 500 |
| Hateful (label=1) | 250 (50%) |
| Not Hateful (label=0) | 250 (50%) |
| Bilder | `data/img/*.png` (500 PNG-Dateien) |

**Bewusste Entscheidung 50/50:** Für den Vergleich von Prompt-Strategien ist eine balancierte Verteilung besser — ein naiver Klassifikator ("immer not hateful") erreicht sonst gratis ~64% Accuracy (natürliche Verteilung im Dataset wäre 36/64).

**JSONL-Format:**
```json
{"id": 8291, "img": "img/08291.png", "label": 1, "text": "white people is this a shooting range"}
```

**Pfad-Ableitung im Code:**  
`dev.jsonl` wird aus `img_folder` abgeleitet: `os.path.dirname(cfg["img_folder"])` → `data/`

---

## 5. Projektstruktur

```
hate_meme_platform/
│
├── hateful_memes_app/              ← Python-Paket
│   ├── app.py                      ← Streamlit-App (Einstiegspunkt)
│   ├── phase1.py                   ← QWEN-Bildbeschreibung (Generator)
│   ├── phase2.py                   ← PHI-4-MINI-Klassifizierung (Generator)
│   ├── experiment_runner.py        ← Automatischer Kombinations-Runner
│   ├── metrics.py                  ← Accuracy, AUROC, F1 berechnen
│   ├── rag.py                      ← RAG-Retriever (ChromaDB)
│   ├── setup_rag.py                ← ChromaDB befüllen
│   ├── finetune.py                 ← QLoRA Fine-Tuning (noch nicht verwendet)
│   ├── excel_utils.py              ← Excel/CSV lesen & schreiben
│   ├── ollama_utils.py             ← Ollama API-Wrapper (call_ollama)
│   ├── config.py                   ← load_config / save_config
│   ├── utils.py                    ← load_jsonl u.a.
│   ├── config.json                 ← Lokale Pfade (nicht im Repo!)
│   ├── requirements.txt
│   └── tests/                      ← 20 pytest-Tests (alle grün)
│
├── data/                           ← Nicht im Repo (.gitignore)
│   ├── img/                        ← 500 PNG-Bilder
│   ├── dev.jsonl                   ← 500 Einträge (50/50)
│   ├── prompts.xlsx                ← Prompt-Definitionen
│   ├── phase1_results.xlsx         ← Phase-1-Ergebnisse (je Sheet = 1 Prompt)
│   └── phase2_results.xlsx         ← Phase-2-Ergebnisse + Metriken-Sheets
│
├── .gitignore
├── README.md                       ← Benutzer-Anleitung
└── projectOverview.md              ← Dieses Dokument
```

---

## 6. Konfiguration (config.json)

Liegt in `hateful_memes_app/config.json` — **nicht im Repo** (maschinespezifisch, in .gitignore).

```json
{
  "prompt_excel":       "C:\\Users\\Leopo\\Claude_Projekte\\hate_meme_platform\\data\\prompts.xlsx",
  "img_folder":         "C:\\Users\\Leopo\\Claude_Projekte\\hate_meme_platform\\data\\img",
  "results_folder":     "C:\\Users\\Leopo\\Claude_Projekte\\hate_meme_platform\\data",
  "max_tokens_phase1":  2500,
  "max_time_seconds":   120,
  "phase1_excel":       "C:\\Users\\Leopo\\Claude_Projekte\\hate_meme_platform\\data\\phase1_results.xlsx",
  "phase2_excel":       "C:\\Users\\Leopo\\Claude_Projekte\\hate_meme_platform\\data\\phase2_results.xlsx",
  "ft_model_path":      ""
}
```

---

## 7. Prompts (prompts.xlsx)

### Sheet "Phase1" — Prompts für QWEN (Bildbeschreibung)

| Name | Strategie |
|------|-----------|
| ZS | Zero-Shot — einfache Bildbeschreibung |
| ZS+RP+AD | + Rollenprompt + Anforderung an Details |
| ZS+RP+CoT+AD | + Chain-of-Thought Schritt-für-Schritt |
| ZS+RP+AD(min) | Kompakt/minimale Variante |

### Sheet "Phase2" — Prompts für PHI-4-MINI (Klassifizierung)

| Name | Strategie |
|------|-----------|
| ZS | Zero-Shot |
| CoT+FS+AD | Chain-of-Thought + Few-Shot + Anweisungen |

**Pflicht-Format in allen Phase-2-Prompts:**
```
Antworte ausschließlich im JSON-Format:
{"reasoning": "...", "label": 0, "confidence": 75}
```
- `label`: 0 = not hateful, 1 = hateful
- `confidence`: Integer 0–100 (wird als Wahrscheinlichkeit für AUROC genutzt)

---

## 8. Experimentenmatrix (vollständig)

**4 Phase-1-Prompts × 6 Phase-2-Prompts = 24 Kombinationen**

Legende: ✅ geplant | 🔒 braucht Fine-Tuned Modell (noch nicht trainiert)

| Phase 1 ↓ \ Phase 2 → | ZS | CoT+FS+AD | CoT+FS+AD+RAG | ZS+RAG | CoT+FS+AD+FT 🔒 | ZS+FT 🔒 |
|------------------------|----|-----------|--------------:|--------|-----------------|----------|
| **ZS** | ✅ | ✅ | ✅ | ✅ | 🔒 | 🔒 |
| **ZS+RP+AD** | ✅ | ✅ | ✅ | ✅ | 🔒 | 🔒 |
| **ZS+RP+CoT+AD** | ✅ | ✅ | ✅ | ✅ | 🔒 | 🔒 |
| **ZS+RP+AD(min)** | ✅ | ✅ | ✅ | ✅ | 🔒 | 🔒 |

**Ohne Fine-Tuning: 16 Kombinationen** (sofort durchführbar)  
**Mit Fine-Tuning: +8 Kombinationen** (später, braucht train.jsonl + QLoRA)

### Was jede Strategie testet

**Phase 1 — Wie detailliert ist die Bildbeschreibung?**
| Prompt | Beschreibung | Erwartung |
|--------|-------------|-----------|
| ZS | Einfachste Beschreibung | Baseline |
| ZS+RP+AD | Rollenprompt + Detailanforderung | Mehr relevante Details |
| ZS+RP+CoT+AD | + Schritt-für-Schritt Reasoning | Strukturiertere Analyse |
| ZS+RP+AD(min) | Kompaktversion | Effizienz ohne Qualitätsverlust? |

**Phase 2 — Wie klassifiziert das LLM?**
| Prompt | Beschreibung | Erwartung |
|--------|-------------|-----------|
| ZS | Nur Aufgabenbeschreibung | Baseline |
| CoT+FS+AD | CoT + Beispiele + Anweisungen | Deutlich besser als ZS |
| CoT+FS+AD+RAG | + ähnliche Fälle aus DB | Kontextualisierung hilft? |
| ZS+RAG | ZS + RAG-Kontext | RAG-Effekt isoliert messen |
| CoT+FS+AD+FT | + Fine-Tuned Modell | Maximale Leistung |
| ZS+FT | ZS + Fine-Tuned Modell | FT-Effekt isoliert messen |

### Excel-Sheet-Namen der Ergebnisse

Jede Kombination erzeugt zwei Sheets in `phase2_results.xlsx`:

| Kombination | Ergebnis-Sheet | Metriken-Sheet |
|-------------|---------------|----------------|
| ZS × ZS | `ZSxZS` | `M_ZSxZS` |
| ZS × CoT+FS+AD | `ZSxCoT+FS+AD` | `M_ZSxCoT+FS+AD` |
| ZS+RP+AD × CoT+FS+AD | `ZS+RP+ADxCoT+FS+AD` | `M_ZS+RP+ADxCoT+FS+AD` |
| ... | ... | ... |

---

## 9. Datenfluss & Checkpoint-Logik

```
Phase 1 läuft
    │
    ├─ Pro Bild: row → append_to_csv(phase1_ZS.csv)   ← Checkpoint
    │
    └─ Am Ende: phase1_ZS.csv → phase1_results.xlsx (Sheet "ZS")
                phase1_ZS.csv wird gelöscht

Phase 2 läuft
    │
    ├─ Pro Eintrag: row → append_to_csv(phase2_ZSxZS.csv)   ← Checkpoint
    │
    └─ Am Ende: phase2_ZSxZS.csv → phase2_results.xlsx (Sheet "ZSxZS")
                Metriken → Sheet "M_ZSxZS"
                phase2_ZSxZS.csv wird gelöscht
```

**Resume-Logik:** Wenn CSV existiert = unterbrochener Lauf → bereits verarbeitete IDs überspringen.  
**Neu starten:** CSV löschen (über "Neu starten"-Button oder manuell).  
**Sheet-Name-Logik:** Gleicher Name = Überschreiben (`if_sheet_exists="replace"`). Kein Timestamp, weil sonst Resume nicht funktioniert.

---

## 10. Excel-Sheet-Benennung

```python
# Phase 1: Sheet-Name = Prompt-Name (max 31 Zeichen)
safe_sheet_name("ZS+RP+AD")          → "ZS+RP+AD"

# Phase 2: Sheet-Name = Phase1-Sheet × Phase2-Prompt
safe_sheet_name("ZSxCoT+FS+AD")      → "ZSxCoT+FS+AD"

# Metriken-Sheet: Präfix "M_"
safe_sheet_name("M_ZSxCoT+FS+AD")    → "M_ZSxCoT+FS+AD"

# Sonderzeichen werden zu "x", max 31 Zeichen
```

---

## 11. Metriken

```python
calculate_metrics(y_true, y_pred, y_prob)
→ {
    "accuracy": 0.72,
    "auroc": 0.7934,
    "f1": 0.701,
    "n_samples": 487
  }
```

**AUROC-Sonderfall:** Bei Single-Class-Input (alle Predictions gleich) gibt sklearn `nan` zurück → wird als `"n/a"` gespeichert (kein Crash).

---

## 12. App starten

```powershell
# Terminal 1: Ollama
ollama serve

# Terminal 2: Streamlit
cd C:\Users\Leopo\Claude_Projekte\hate_meme_platform\hateful_memes_app
streamlit run app.py --server.port 8501
```

Browser: `http://localhost:8501`

---

## 13. Bekannte Eigenheiten & Fixes (bereits implementiert)

| Problem | Ursache | Fix |
|---------|---------|-----|
| `PermissionError` beim Excel-Schreiben | Excel war geöffnet (Windows-Datei-Lock) | Excel schließen; CSV-Checkpoint überlebt, Resume möglich |
| Log-Duplikation im UI | `st.container()` akkumuliert, schreibt 1+2+3... Einträge | `st.empty()` + `container.markdown()` ersetzt Inhalt |
| Modell nicht gefunden (404) | `qwen2.5-vl:3b` (mit Bindestrich) ≠ `qwen2.5vl:3b` | Modellname in phase1.py korrigiert |
| `nan` AUROC | sklearn gibt nan statt ValueError bei Single-Class | `math.isnan()`-Check in metrics.py |
| Windows-Filelocks in Tests | ChromaDB hält SQLite-Lock auf temp dir | `ignore_cleanup_errors=True` in TemporaryDirectory |
| macOS-Stub-Dateien (`._*.png`) | macOS-Metadaten-Dateien im img-Ordner | `not f.startswith('._')` beim Filtern |

---

## 14. Git-Status

**Repo:** `C:\Users\Leopo\Claude_Projekte\hate_meme_platform`  
**Branch:** `master`  
**Status:** Sauber (kein uncommittetes Material)

**Wichtigste Commits:**
```
d161334  chore: ignore PFT.pptx presentation file
57b5b68  chore: add .gitignore for build artifacts and local files
30b6453  fix: correct Ollama model name and log duplication in UI
54bdcb8  fix: resolve all test failures — Windows file lock and nan AUROC
a335416  fix: 5 bugs from code review
```

**In .gitignore (nicht im Repo):**
- `data/` — Bilder, Ergebnisse, JSONL (Urheberrecht + Größe)
- `hateful_memes_app/config.json` — absolute Pfade
- `chroma_db/` — RAG-Datenbank
- `PFT.pptx` — Präsentation
- `__pycache__/`, `*.log`, etc.

---

## 15. Tests

```powershell
cd hateful_memes_app
pytest tests/ -v
# → 20/20 Tests grün
```

Getestete Module: config, utils, excel_utils, metrics, rag, phase_logic

---

## 15. Offene Themen / Nächste Schritte

- [ ] Phase 1 für alle 500 Bilder durchlaufen lassen (dauert 2–4 Stunden)
- [ ] Phase 2 mit verschiedenen Prompt-Kombinationen auswerten
- [ ] RAG evaluieren (setup_rag.py erst ausführen wenn Phase 1 fertig)
- [ ] Fine-Tuning (finetune.py) — noch nicht gestartet, braucht train.jsonl

---

## 16. Nützliche Befehle

```powershell
# Ollama-Status prüfen
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps

# Welche Modelle sind installiert?
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list

# Tests ausführen
cd C:\Users\Leopo\Claude_Projekte\hate_meme_platform\hateful_memes_app
& "C:\Users\Leopo\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/ -v

# Git-Status
git -C "C:\Users\Leopo\Claude_Projekte\hate_meme_platform" log --oneline -5
```

# Hateful Memes Classification Platform

Zweistufige KI-Pipeline zur Klassifizierung von Hate-Speech in Memes.  
Entwickelt als wissenschaftliches Experiment zum Vergleich verschiedener Prompt-Strategien.

---

## Inhaltsverzeichnis

1. [Projektübersicht](#projektübersicht)
2. [Architektur](#architektur)
3. [Voraussetzungen & Installation](#voraussetzungen--installation)
4. [Erste Schritte](#erste-schritte)
5. [App starten](#app-starten)
6. [Tab-Übersicht](#tab-übersicht)
7. [Phase 1 — Bildbeschreibung](#phase-1--bildbeschreibung)
8. [Phase 2 — Klassifizierung](#phase-2--klassifizierung)
9. [Experiment-Runner](#experiment-runner)
10. [Excel-Dateien](#excel-dateien)
11. [Datensatz](#datensatz)
12. [Wichtige Hinweise](#wichtige-hinweise)
13. [Projektstruktur](#projektstruktur)

---

## Projektübersicht

Die App implementiert eine **zweistufige Pipeline**:

| Stufe | Modell | Aufgabe |
|-------|--------|---------|
| Phase 1 | QWEN2.5-VL-3B (multimodal) | Bild analysieren → strukturierte Textbeschreibung |
| Phase 2 | PHI-4-MINI (Sprachmodell) | Textbeschreibung + Meme-Text → Hate/Not Hate |

**Ziel:** Verschiedene Prompt-Strategien (Zero-Shot, Chain-of-Thought, Few-Shot, RAG, Fine-Tuning) systematisch vergleichen und Metriken (Accuracy, AUROC, F1) auswerten.

**Datensatz:** [Hateful Memes Dataset](https://ai.meta.com/tools/hatefulmemes/) von Meta AI  
500 Bilder, perfekt balanciert: **250 hateful / 250 not hateful (50/50)**

---

## Architektur

```
Bild (PNG)
    │
    ▼
┌─────────────────────────────────┐
│  Phase 1: QWEN2.5-VL-3B        │
│  "Beschreibe dieses Meme..."    │
│  → strukturierte Beschreibung   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Phase 2: PHI-4-MINI            │
│  Beschreibung + Meme-Text       │
│  → {"label": 0/1,               │
│     "confidence": 85,           │
│     "reasoning": "..."}         │
└────────────────┬────────────────┘
                 │
                 ▼
         Excel-Ergebnisse
         + Metriken (Acc, AUROC, F1)
```

---

## Voraussetzungen & Installation

### System
- Windows 10/11
- Python 3.12
- [Ollama](https://ollama.com) (lokal installiert)

### KI-Modelle (einmalig herunterladen)
```powershell
ollama pull qwen2.5vl:3b
ollama pull phi4-mini:latest
```

### Python-Abhängigkeiten
```powershell
cd C:\Users\Leopo\Claude_Projekte\hate_meme_platform\hateful_memes_app
pip install -r requirements.txt
```

---

## Erste Schritte

### 1. Ordnerstruktur prüfen

Nach dem Setup sollte `data/` so aussehen:
```
data/
├── img/                    ← 500 PNG-Bilder (vom Desktop kopiert)
├── dev.jsonl               ← 500 Einträge (250 hate / 250 not hate)
├── prompts.xlsx            ← Prompt-Definitionen (Phase 1 + Phase 2)
├── phase1_results.xlsx     ← Ergebnisse Phase 1 (wird befüllt)
└── phase2_results.xlsx     ← Ergebnisse Phase 2 (wird befüllt)
```

### 2. config.json prüfen

Die Datei `hateful_memes_app/config.json` enthält alle Pfade.  
Sie ist in `.gitignore` (nicht im Repo) und muss lokal vorhanden sein.

```json
{
  "prompt_excel":       "C:\\...\\data\\prompts.xlsx",
  "img_folder":         "C:\\...\\data\\img",
  "results_folder":     "C:\\...\\data",
  "max_tokens_phase1":  2500,
  "max_time_seconds":   120,
  "phase1_excel":       "C:\\...\\data\\phase1_results.xlsx",
  "phase2_excel":       "C:\\...\\data\\phase2_results.xlsx",
  "ft_model_path":      ""
}
```

---

## App starten

### Schritt 1: Ollama starten
```powershell
ollama serve
```
> Ollama muss laufen, bevor die App gestartet wird.  
> Fenster offen lassen (läuft im Hintergrund).

### Schritt 2: Streamlit starten
```powershell
cd C:\Users\Leopo\Claude_Projekte\hate_meme_platform\hateful_memes_app
streamlit run app.py --server.port 8501
```

### Schritt 3: Browser öffnen
```
http://localhost:8501
```

---

## Tab-Übersicht

| Tab | Zweck |
|-----|-------|
| ⚙️ Einstellungen | Pfade und Parameter konfigurieren |
| 📷 Phase 1 | Bildbeschreibungen generieren (QWEN) |
| 🧠 Phase 2 | Klassifizierung auswerten (PHI-4-MINI) |
| 🚀 Experiment-Runner | Alle Prompt-Kombinationen automatisch durchlaufen |

---

## Phase 1 — Bildbeschreibung

**Modell:** QWEN2.5-VL-3B (multimodal, sieht das Bild)

**Was passiert:**
1. Alle 500 Bilder aus `data/img/` werden nacheinander verarbeitet
2. QWEN beschreibt jedes Bild anhand des gewählten Prompts
3. Ergebnisse werden als CSV gespeichert (Checkpoint)
4. Am Ende: CSV → Excel-Sheet (ein Sheet pro Prompt)

**Prompt-Strategien (Sheet "Phase1" in prompts.xlsx):**
| Name | Beschreibung |
|------|-------------|
| ZS | Zero-Shot — einfache Beschreibung |
| ZS+RP+AD | + Rollenbeschreibung + Aufforderung zu Details |
| ZS+RP+CoT+AD | + Chain-of-Thought Reasoning |
| ZS+RP+AD(min) | Kompakte Variante |

**Resume-Funktion:**
- Wird ein Lauf unterbrochen (z.B. Absturz, manuell gestoppt), erkennt die App das automatisch
- Beim nächsten Start: "Unterbrochener Lauf: X Einträge bereits verarbeitet" → **Fortsetzen** anhaken
- Die CSV-Datei dient als Checkpoint und wird nach erfolgreichem Abschluss gelöscht

**Dauer:** ca. 2–6 Stunden für 500 Bilder (abhängig von Hardware)

---

## Phase 2 — Klassifizierung

**Modell:** PHI-4-MINI (reines Sprachmodell, sieht kein Bild)

**Eingabe:** Bildbeschreibung (aus Phase 1) + originaler Meme-Text  
**Ausgabe:** `{"reasoning": "...", "label": 0, "confidence": 75}`

**Prompt-Strategien (Sheet "Phase2" in prompts.xlsx):**
| Name | Beschreibung |
|------|-------------|
| ZS | Zero-Shot |
| CoT+FS+AD | Chain-of-Thought + Few-Shot + Anweisungen |

**Optionen:**
- **RAG:** Ähnliche Beispiele aus der Datenbank als Kontext
- **Fine-Tuned Modell:** Eigenes feingetuntes Modell verwenden (Pfad in Einstellungen)

**Metriken:** Accuracy, AUROC, F1-Score werden direkt angezeigt

---

## Experiment-Runner

Führt **alle gewählten Phase-1 × Phase-2 Kombinationen** automatisch durch.

**Beispiel:** 4 Phase-1 Prompts × 2 Phase-2 Prompts = **8 Kombinationen**

- Überspringe bereits vorhandene Kombinationen automatisch (Resume)
- Zeigt Fortschritt pro Kombination + Gesamt
- Am Ende: `🎉` und alle Metriken in phase2_results.xlsx

---

## Excel-Dateien

### prompts.xlsx
Enthält die Prompt-Definitionen. Zwei Sheets:
- **Phase1** — Prompts für QWEN (Bildbeschreibung)
- **Phase2** — Prompts für PHI-4-MINI (Klassifizierung)

Jede Zeile: `Prompt-Name | Prompt-Text`

### phase1_results.xlsx
Ein Sheet pro Prompt-Name. Spalten:
`id | img | text | description | prompt_name | status`

### phase2_results.xlsx
Ein Sheet pro Kombination (z.B. "ZS × CoT+FS+AD"). Spalten:
`id | img | text | label_true | label_pred | confidence | reasoning | status`

> **Wichtig:** Gleicher Sheet-Name = Überschreiben (kein Duplikat-Problem).  
> Das ermöglicht Re-Runs einzelner Kombinationen.

---

## Datensatz

**Hateful Memes Dataset** (Meta AI Research, 2020)

| Eigenschaft | Wert |
|-------------|------|
| Datei | `data/dev.jsonl` |
| Einträge | 500 |
| Hateful (label=1) | 250 (50%) |
| Not Hateful (label=0) | 250 (50%) |
| Bilder | `data/img/*.png` |

**JSONL-Format:**
```json
{"id": 8291, "img": "img/08291.png", "label": 1, "text": "some meme text"}
```

> `data/` ist in `.gitignore` — Bilder und Ergebnisse werden nicht ins Repo eingecheckt (Urheberrecht + Größe).

---

## Externe Abhängigkeiten

**HatRed** (VL-T5 / Hateful-Meme-Reasoning) — fremdes Forschungs-Repo, das hier **nicht** mit eingecheckt wird (eigene Lizenz, eigene Datensätze).

| | |
|---|---|
| Quelle | <https://github.com/Social-AI-Studio/HatRed> |
| Verwendeter Stand | Commit `3ee5e59` |
| Lokaler Pfad | `HatRed/` (in `.gitignore`) |

Bei Bedarf separat klonen:
```bash
git clone https://github.com/Social-AI-Studio/HatRed.git
```

---

## Wichtige Hinweise

### Excel muss geschlossen sein
Wenn Phase 1 oder Phase 2 fertig ist und die Ergebnisse in die Excel schreibt, darf die Datei **nicht in Excel geöffnet sein** — sonst:
```
PermissionError: [Errno 13] Permission denied: 'phase1_results.xlsx'
```
**Lösung:** Excel schließen → App erneut starten → Resume nutzen (CSV-Checkpoint ist noch da).

### Ollama muss vor der App gestartet sein
```powershell
ollama serve   # erst dann:
streamlit run app.py
```

### Modellnamen (exakt so in Ollama)
```
qwen2.5vl:3b       ← kein Bindestrich!
phi4-mini:latest
```

### config.json ist nicht im Repo
Enthält absolute Pfade (maschinespezifisch). Bei Neuinstallation auf einem anderen Rechner muss `config.json` manuell angelegt oder über die App im Tab "Einstellungen" konfiguriert werden.

---

## Projektstruktur

```
hate_meme_platform/
│
├── hateful_memes_app/
│   ├── app.py                  # Streamlit-App (4 Tabs)
│   ├── phase1.py               # Phase 1: QWEN-Bildbeschreibung
│   ├── phase2.py               # Phase 2: PHI-4-MINI-Klassifizierung
│   ├── experiment_runner.py    # Automatischer Kombinations-Runner
│   ├── metrics.py              # Accuracy, AUROC, F1
│   ├── rag.py                  # RAG-Retriever (ChromaDB)
│   ├── setup_rag.py            # RAG-Datenbank befüllen
│   ├── finetune.py             # QLoRA Fine-Tuning (optional)
│   ├── excel_utils.py          # Excel lesen/schreiben
│   ├── ollama_utils.py         # Ollama API-Wrapper
│   ├── config.py               # Konfiguration laden/speichern
│   ├── utils.py                # JSONL laden u.a.
│   ├── config.json             # Lokale Pfade (nicht im Repo!)
│   ├── requirements.txt        # Python-Abhängigkeiten
│   └── tests/                  # pytest-Tests (20 Tests)
│
├── data/                       # Nicht im Repo (.gitignore)
│   ├── img/                    # 500 PNG-Bilder
│   ├── dev.jsonl               # 500 Einträge (50/50)
│   ├── prompts.xlsx            # Prompt-Definitionen
│   ├── phase1_results.xlsx     # Phase-1-Ergebnisse
│   └── phase2_results.xlsx     # Phase-2-Ergebnisse
│
├── .gitignore
└── README.md                   # Diese Datei
```

---

## Tests ausführen

```powershell
cd hateful_memes_app
pytest tests/ -v
```

Alle 20 Tests sollten grün sein.

---

*Erstellt im Rahmen einer wissenschaftlichen Arbeit zum Thema multimodale Hate-Speech-Erkennung.*

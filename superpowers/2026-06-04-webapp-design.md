# Web-App Design – Hassrede in Online Memes

**Datum:** 2026-06-04  
**Projekt:** Prompting & Fine-Tuning (PFT), SoSe 2026  
**Autor:** Leopold Wittmann  
**Betreuer:** Prof. Jan Vellmer

---

## Überblick

Streamlit-basierte Web-App zur Steuerung und Live-Inspektion der zweiphasigen Hate-Speech-Klassifizierungspipeline. Die App läuft lokal auf dem Gaming-Laptop, spricht Ollama für Inferenz an und schreibt alle Ergebnisse in Excel-Dateien.

---

## Architektur

```
app.py                   ← Streamlit-Einstiegspunkt, Tab-Navigation
├── phase1.py            ← QWEN2.5-VL-3B via Ollama, Bildbeschreibungen
├── phase2.py            ← PHI-4-MINI via Ollama, Klassifizierung
├── experiment_runner.py ← Iteriert gewählte Kombinationen per Checkbox
├── metrics.py           ← Accuracy, AUROC, Precision, Recall, F1, Confusion Matrix
├── rag.py               ← ChromaDB Setup und Retrieval
├── ollama_utils.py      ← Gemeinsamer call_ollama() für QWEN und PHI-4-MINI
├── utils.py             ← load_jsonl() und weitere Hilfsfunktionen
└── config.py            ← Lesen/Schreiben config.json
```

**Externe Dienste:**
- Ollama (lokal, Python-Client `ollama`)
- Hugging Face PEFT / QLoRA (separates Skript, nicht über App gesteuert)
- ChromaDB (lokal, kein Server)

---

## Tabs

### Tab 1: Einstellungen

Konfigurierbare Felder, gespeichert in config.json:

| Feld | Beschreibung |
|------|-------------|
| Prompt-Excel | Pfad zur Excel-Datei mit allen Prompts |
| Bild-Ordner | Pfad zum img/-Ordner des Datensatzes |
| Ergebnis-Ordner | Zentraler Ordner für alle Outputs |
| Max. Token (Phase 1) | Maximale Token pro Bildbeschreibung; bei Überschreitung wird das Bild übersprungen |
| Max. Zeit pro Bild (Sek.) | Maximale Wartezeit pro Bild in Sekunden; bei Überschreitung wird das Bild übersprungen und als "timeout" markiert |
| Phase-1-Ergebnis-Excel | Eine Excel-Datei für alle Phase-1-Ergebnisse; jeder Prompt-Lauf bekommt ein eigenes Sheet (z.B. "ZS", "ZS+RP+AD") |
| Phase-2-Ergebnis-Excel | Eine Excel-Datei für alle Phase-2-Ergebnisse; jede Kombination bekommt ein eigenes Sheet |
| Fine-Tuned Modell (Pfad) | Ollama-Modellname des QLoRA-Adapters (z.B. phi4-mini-ft); leer lassen bis Fine-Tuning abgeschlossen |

Speichern-Button schreibt alle Werte in config.json.

---

### Tab 2: Phase 1

**Ziel:** Alle Bilder aus img/ durch QWEN2.5-VL-3B laufen lassen und Bildbeschreibungen in Excel speichern.

**UI-Elemente:**
- Radio-Button: Modus (Experiment-Modus = dev.jsonl ~500 Bilder / Fine-Tuning-Modus = train.jsonl 8500 Bilder)
- Dropdown: Prompt-Name (aus Prompt-Excel, Sheet "Phase1")
- Textfeld: Prompt-Text (editierbar, vorbelegt durch Dropdown-Auswahl)
- Start-Button
- Fortschrittsbalken (dynamisch, passt sich an tatsächliche Bildanzahl an)
- Scrollender Log (neuester Eintrag oben): ID | Meme-Text | vollständige Bildbeschreibung | Status

**Verhalten:**
- Beim Start: prüft ob eine CSV aus einem unterbrochenen Lauf existiert (phase1_PROMPTNAME.csv im Ergebnis-Ordner)
- Falls vorhanden: Info "X Einträge bereits verarbeitet", Checkbox "Fortsetzen" + Button "Neu starten"
- Frischer Start löscht die alte CSV; Resume liest verarbeitete IDs aus der CSV
- Während des Laufs: jeder Eintrag sofort in CSV geschrieben (kein Datenverlust bei Absturz)
- Nach Abschluss: CSV wird in Phase-1-Ergebnis-Excel exportiert (Sheet = Prompt-Name), CSV wird gelöscht
- Wenn Max-Token überschritten: Eintrag mit Status "token_limit" markieren, zum nächsten Bild springen
- Wenn Max-Zeit überschritten: Eintrag mit Status "timeout" markieren, zum nächsten Bild springen
- Excel-Spalten: id, img, text (Meme-Text aus JSONL), description (Bildbeschreibung), prompt_name, status

---

### Tab 3: Phase 2

Zwei Modi, wählbar per Radio-Button.

#### Modus A: Single Run

**Ziel:** Einen einzelnen Phase-2-Lauf starten – primär für die Live-Demo.

**UI-Elemente:**
- Dropdown: Phase-1-Sheet auswählen (aus den vorhandenen Sheets der Phase-1-Ergebnis-Excel)
- Dropdown: Basis-Prompt (aus Prompt-Excel, Sheet "Phase2") – ZS oder CoT+FS+AD
- Textfeld: Prompt-Text (editierbar)
- Checkbox: RAG aktivieren (hängt Top-3 ChromaDB-Chunks automatisch an den Prompt)
- Checkbox: Fine-Tuned Modell verwenden (statt PHI-4-MINI base das QLoRA-Modell nutzen)
- Start-Button
- Fortschrittsbalken (dynamisch)
- Scrollender Log: ID | Meme-Text | Label (0/1) | Confidence | vollständiges Reasoning
- Nach Abschluss: Metriken automatisch als neues Sheet in Phase-2-Ergebnis-Excel schreiben (Sheet-Name = Kombination + Timestamp)

#### Modus B: Experiment-Runner

**Ziel:** Frei wählbare Kombinationen der Experiment-Matrix automatisch durchlaufen.

**UI-Elemente:**

Kontrollkästchen Phase 1 (Mehrfachauswahl):
- ZS
- ZS+RP+AD
- ZS+RP+CoT+AD
- ZS+RP+AD(min)

Kontrollkästchen Phase 2 (Mehrfachauswahl):
- ZS
- CoT+FS+AD
- CoT+FS+AD+RAG
- CoT+FS+AD+FT
- ZS+RAG
- ZS+FT

Anzeige: "X Kombinationen ausgewählt"  
Start-Button  
Aktuelle Kombination: "Kombination 7/12: ZS+RP+AD × CoT+FS+AD"  
Fortschrittsbalken pro Kombination + Gesamtfortschritt  
Scrollender Log (wie Single Run)  
Nach jeder Kombination: Metriken als neues Sheet in Phase-2-Ergebnis-Excel schreiben (Sheet-Name = Kombination, z.B. "ZS+RP+AD×CoT+FS+AD")

**Hinweis zu RAG/FT im Runner:** CoT+FS+AD+RAG = CoT+FS+AD-Prompt mit RAG aktiv. ZS+FT = ZS-Prompt mit Fine-Tuned Modell. Der Runner steuert das intern automatisch.

**Checkpoint:**
- checkpoint_runner.json speichert welche Kombinationen abgeschlossen sind und die letzte verarbeitete ID der aktuellen Kombination
- Beim Neustart: abgeschlossene überspringen, aktuelle ab letzter ID fortsetzen

---

## Prompt-Excel Format

Zwei Sheets.

**Sheet "Phase1"**

| Name | Prompt |
|------|--------|
| ZS | Beschreibe, was auf dem Bild zu sehen ist. |
| ZS+RP+AD | Du bist ein auf Bildbeschreibung spezialisierter Algorithmus. Beschreibe das Bild mit Fokus auf hate-speech-relevante Attribute... |
| ZS+RP+CoT+AD | Wie ZS+RP+AD, zusätzlich mit Chain-of-Thought-Anweisung |
| ZS+RP+AD(min) | Wie ZS+RP+AD, aber mit minimierter Attributliste |

**Sheet "Phase2"**

| Name | Prompt |
|------|--------|
| ZS | Klassifiziere, ob dieses Meme hateful ist. |
| CoT+FS+AD | Du bist ein hochpräziser Algorithmus zur Klassifizierung von Hass in Memes. Denke schrittweise nach... (mit Few-Shot-Beispielen und Attributliste) |

**Wichtig:** Jeder Phase-2-Prompt muss die JSON-Format-Instruktion enthalten: confidence als Integer 0-100, Antwort ausschließlich als JSON-Objekt {"reasoning": "...", "label": 0, "confidence": 75}.

---

## Datenfluss

**Phase 1:**
dev.jsonl (Experiment) oder train.jsonl (Fine-Tuning) → ID + Bildpfad → Prompt + Max-Token/Zeit-Limit → QWEN2.5-VL-3B via ollama_utils.call_ollama() → Bildbeschreibung → sofort in CSV schreiben → nach Abschluss: CSV → Excel-Sheet, CSV löschen

**Phase 2:**
Phase-1-Excel (Experiment-Modus-Sheet) + dev.jsonl → ID + Bildbeschreibung + Meme-Text → [wenn RAG aktiv] ChromaDB Top-3 Chunks anhängen → Basis-Prompt → PHI-4-MINI via ollama_utils.call_ollama() ODER Fine-Tuned Modell → JSON {label, confidence, reasoning} → sofort in CSV schreiben → nach Abschluss: CSV → Excel-Sheet, Metriken-Sheet, CSV löschen

**Metriken:** Accuracy, AUROC, Precision, Recall, F1, Confusion Matrix (TP/FP/TN/FN) via sklearn.metrics

---

## Fehlerbehandlung

| Situation | Verhalten |
|-----------|-----------|
| Ollama nicht erreichbar | Fehlermeldung im Log, Lauf stoppt |
| Kein valides JSON vom Modell | Eintrag als "error" markieren, überspringen, weiterlaufen |
| Max-Token überschritten | Eintrag als "token_limit" markieren, nächstes Bild |
| Max-Zeit überschritten | Eintrag als "timeout" markieren, nächstes Bild |
| CSV nicht schreibbar | Fehlermeldung im Log, Lauf stoppt |
| Absturz / Prozess unterbrochen | Checkpoint greift beim nächsten Start |

---

## Checkpoint-Mechanismus

Kein separater Checkpoint-JSON. Die CSV-Datei ist der Checkpoint.

**Phase 1:** phase1_PROMPTNAME.csv im Ergebnis-Ordner enthält alle verarbeiteten Zeilen. Beim Resume: id-Spalte einlesen = verarbeitete IDs.

**Phase 2:** phase2_SHEETNAME.csv analog.

**Experiment-Runner:** runner_completed.csv mit einer Zeile pro abgeschlossener Kombination.

**Frischer Start:** CSV löschen. **Nach Erfolg:** CSV wird automatisch gelöscht.
---

## Abhängigkeiten

```
streamlit
pandas
openpyxl
ollama                 # Offizieller Python-Client für Ollama
chromadb               # Lokaler Vector Store für RAG
sentence-transformers  # Embedding-Modell für ChromaDB (~500 MB Download beim ersten Start)
scikit-learn           # Metriken (AUROC, F1 etc.)
pytest                 # Unit-Tests
```

---

## Nicht im Scope

- Fine-Tuning über die App starten (separates Skript)
- Benutzerverwaltung oder Auth
- Deployment außerhalb localhost
- Datenbank (alles Excel-basiert)

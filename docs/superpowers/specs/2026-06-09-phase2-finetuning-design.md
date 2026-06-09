# Phase-2 QLoRA Fine-Tuning — Design / Spec

> Status: **Entwurf zur Review**. Datum: 2026-06-09.
> Vorgänger-Kontext: `projectOverview.md`, `docs/superpowers/plans/2026-06-08-runpod-phase1-deployment.md`.

## 1. Ziel & Erfolgskriterium

Den **Phase-2-Klassifikator** (`phi4-mini` = Phi-4-mini-instruct) per **4-bit QLoRA** fine-tunen, das Ergebnis als **Ollama-Modell** bereitstellen und über `ft_model_path` in die bestehende Pipeline einhängen.

**Zweck:** die Matrix-Zellen **`ZS+FT`** und **`CoT+FS+AD+FT`** füllen und den Fine-Tuning-Effekt isoliert messen (FT vs. Nicht-FT bei sonst gleichem Phase-1-/Phase-2-Setup), gemessen auf dev (500) an **Accuracy, AUROC, F1**.

**Definition of Done:**
1. `ollama list` zeigt ein lauffähiges Modell `phase2-ft`.
2. Die bestehende Phase-2-Pipeline produziert damit auf dev gültige Metriken für `ZS+FT` und `CoT+FS+AD+FT`.
3. Die vier Vergleichsläufe auf **denselben `ZS + RP`-dev-Beschreibungen** liegen vor: `ZS`, `ZS+FT`, `CoT+FS+AD`, `CoT+FS+AD+FT` → FT-Effekt ablesbar.

## 2. Datenbasis (vorhanden & verifiziert)

| Artefakt | Pfad | Status |
|---|---|---|
| Train-Beschreibungen (Prosa, `ZS + RP`) | `data/phase1_train_zsrp.xlsx`, Sheet `ZS + RP` | 2000/2000 `ok` ✅ |
| Dev-Beschreibungen (Prosa, `ZS + RP`) | `data/phase1_dev_zsrp.xlsx`, Sheet `ZS + RP` | 500/500 `ok` ✅ |
| Train-Labels | `data/train.jsonl` | 1000/1000 (50/50) ✅ |
| Dev-Labels | `data/dev.jsonl` | 500 ✅ |
| HatReD-Reasoning (hateful) | `HatRed/datasets/hatred/annotations/fhm_train_reasonings.jsonl` | deckt **837** unserer 1000 hateful train-Memes ✅ |
| HatReD-Race (auxiliary) | `HatRed/datasets/hatred/auxiliary/fhm_train_race.jsonl` | dieselben 837 IDs, reichert Reasoning an ✅ |
| Basismodell | Ollama `phi4-mini:3.8b`; Unsloth `unsloth/Phi-4-mini-instruct` | ✅ |

**HatReD-Abdeckung (verifiziert):** Von unseren 2000 train-Memes erscheinen **908** in HatReD — davon **837 hateful** (FHM-Label=1) und **71 benign** (FHM-Label=0, sog. „benign confounders"). Die Race-Datei deckt exakt dieselben 908 IDs ab (0 zusätzliche). Daraus folgt das Trainingsset:
- **837 hateful** mit echtem HatReD-Reasoning (+ Race-Anreicherung im `Targets:`-Präfix)
- **163 hateful ohne** HatReD-Reasoning → **aus Training ausgeschlossen** (kein generisches Fallback-Template)
- **1000 benign** mit benignem Template (inkl. der 71 Konflikt-IDs, die FHM-treu als benign gelernt werden)
- **= 1837 Trainingsbeispiele** (837/1000, leichte Klassenschieflage, von AUROC/F1 toleriert)

**Leakage-Status:** `train ∩ dev = 0`, `train ∩ test = 0`. HatReD-Reasoning/-Race werden nur für **train** verwendet. Sauber.

`ZS + RP`-Prompttext (Phase 1, aus `…_2000/data/prompts.xlsx`):
> „You are a forensic image analyst specializing in hate speech and content moderation research. Describe what you see in this image. Ignore text on the image."

## 3. Zu bestätigende Entscheidungen (Review-Gate)

### E1 — Trainings-/Inferenz-Format (NEU, betrifft phase2.py)
Beide Phase-2-Prompts enthalten `{text}`/`{description}`, werden in [phase2.py](../../../hateful_memes_app/phase2.py) aber **nicht** gefüllt; stattdessen werden Beschreibung+Text separat & deutsch im User-Turn angehängt → Platzhalter bleiben wörtlich stehen, Inhalt doppelt.

- **Option B (empfohlen):** phase2.py so fixen, dass der Prompt **als ein gefüllter User-Turn** genutzt wird (`prompt_text.format(text=…, description=…)`), kein separater deutscher User-Block. Trainingsdaten exakt in diesem Format. Sauber, beseitigt den Platzhalter-Bug. Kostet: kleiner phase2.py-Refactor + die ohnehin nötigen Nicht-FT-Neuläufe auf `ZS + RP`-dev.
- **Option A (Fallback):** aktuelles (fehlerhaftes) Inferenzverhalten 1:1 replizieren. Weniger Änderung, aber der Bug bleibt im Datensatz „eingebrannt".

### E2 — System-/Prompt-Abdeckung im Training
Das **eine** FT-Modell wird unter **zwei** Phase-2-Prompts evaluiert (`ZS+FT`, `CoT+FS+AD+FT`).
- **Empfohlen:** jedes train-Beispiel unter **beiden** Prompt-Templates erzeugen (~2×2000 = 4000 Beispiele) → robust für beide Zellen.
- Alternative: nur ein Template (~2000) oder zwei getrennte FT-Modelle (rigoroser, aber doppelter Trainings-/Verwaltungsaufwand).

### E3 — Export/Eval-Ort
- **Empfohlen:** Training + GGUF-Export auf **RunPod**; `.gguf` per rsync zurück auf den Mac; `ollama create` + dev-Eval **lokal** (M2, phi4-mini ~2.5 GB reicht für Inferenz). Alternative: alles auf RunPod.

## 4. Architektur — 4 Komponenten

```
[1] finetune.py / build_finetune_data() (lokal)
      train-xlsx + train.jsonl + HatReD-Reasoning + HatReD-Race + prompts.xlsx (ZS+RP+AD)
      → data/finetune_data.jsonl   (Chat-Format, 1837 Zeilen)
            │
            ▼  rsync
[2] train_qlora.py (RunPod, Unsloth)
      unsloth/Phi-4-mini-instruct (4-bit) + QLoRA
      → save_pretrained_gguf (q4_k_m) + Modelfile
            │
            ▼  rsync .gguf zurück
[3] ollama create phase2-ft -f Modelfile   (lokal)
      → config: ft_model_path = "phase2-ft"
            │
            ▼
[4] Eval (bestehende Pipeline, phase1_excel = data/phase1_dev_zsrp.xlsx)
      2 Läufe auf ZS+RP-dev: ZS+RP+AD, ZS+RP+AD+FT
      → phase2_results.xlsx + Metriken
```

### Komponente 1 — `hateful_memes_app/finetune.py` (Rewrite: `build_finetune_data`)
Liest Train-Beschreibungen (Sheet `ZS + RP`, nur `status==ok`), Labels aus `train.jsonl`, HatReD-Reasonings, HatReD-Race, den `ZS+RP+AD`-Systemprompt. Erzeugt pro behaltenem Meme **ein** Chat-Beispiel (`system`/`user`/`assistant`).

**Output `assistant`-JSON** (matcht `_parse_json_response` in phase2.py):
```json
{"reasoning": "<s.u.>", "label": <0|1>, "confidence": <int>}
```
- `label`: Ground Truth aus `train.jsonl`.
- `reasoning`:
  - `label==1` & HatReD vorhanden → `Targets: <target> (<race>). ` + `" ".join(reasonings)` (echte menschliche Begründung, Race optional zur Anreicherung).
  - `label==1` **ohne** HatReD → **Beispiel wird übersprungen** (nicht ins Trainingsset).
  - `label==0` → benignes Template (EN: „No attack on a protected group is present. …").
- `confidence` = **P(hateful) in %** (AUROC-Fix): `label==1` → zufällig in `[82,96]`, `label==0` → in `[4,18]`, fester Seed. Kein konstanter Wert → AUROC-Signal bleibt erhalten.

**Format** (Systemprompt-Trennung):
- `system` = `ZS+RP+AD`-Prompttext aus `prompts.xlsx` (reine Instruktion, keine Platzhalter)
- `user` = `f"Meme text: {meme_text}\n\nImage description: {description}"`
- `assistant` = JSON-String

Sprache durchgehend **Englisch** (Beschreibungen, Prompts, HatReD sind EN — das alte deutsche Template in finetune.py ist veraltet und wird ersetzt).

**Tests** (`tests/test_finetune.py`, pytest, fügt sich in die bestehende Suite ein):
- Zeilenzahl = (Beispiele mit Label) × (#Templates).
- Jede `assistant`-Antwort ist gültiges JSON, `label∈{0,1}`, `confidence` int 0–100.
- confidence-Vorzeichen korrekt (hateful hoch, benign niedrig).
- hateful+abgedeckt nutzt HatReD-Text; benign nutzt benignes Template.
- `user`-Inhalt entspricht exakt dem gefüllten Prompt-Template.

### Komponente 2 — `hateful_memes_app/train_qlora.py` (neu, RunPod)
- `FastLanguageModel.from_pretrained("unsloth/Phi-4-mini-instruct", load_in_4bit=True, max_seq_length=4096)` (4096 wegen langer Beschreibungen + Few-Shot-Prompt).
- LoRA: `r=16, lora_alpha=16, lora_dropout=0, target_modules=[q,k,v,o,gate,up,down]_proj`, `use_gradient_checkpointing="unsloth"`.
- Chat-Template des Modells anwenden; **`train_on_responses_only`** (Loss nur auf der Assistant-Antwort).
- Hyperparam (Start): `epochs=2`, `lr=2e-4`, `bs=2 × grad_accum=4`, `warmup_ratio=0.03`, cosine, `bf16`, `optim=adamw_8bit`, fester Seed.
- Export: `model.save_pretrained_gguf("phase2-ft-gguf", tokenizer, quantization_method="q4_k_m")` → GGUF + Modelfile.

### Komponente 3 — RunPod-Setup `hateful_memes_app/launch_runpod_ft.sh` (neu) + Schritt-Doku
- Pod: RunPod-PyTorch-Template, RTX 3090/4090, CUDA 12, ~50 GB Disk.
- `pip install unsloth` (+ Deps); llama.cpp-Toolchain für GGUF (Unsloth zieht/baut sie — **fehleranfällig**, Puffer einplanen).
- Upload `finetune_data.jsonl` + `train_qlora.py`; Lauf; `.gguf` zurück per rsync.
- Lokal: `ollama create phase2-ft -f Modelfile`; `ft_model_path="phase2-ft"`.

### Komponente 4 — Eval (bestehende Pipeline, minimal angepasst)
- `phase1_excel` → `data/phase1_dev_zsrp.xlsx`, `phase1_sheet="ZS + RP"`, `dev_jsonl` → `data/dev.jsonl`.
- **Vier Läufe** auf denselben dev-Beschreibungen für faire FT-Isolation:
  | Lauf | prompt_name | use_ft | model |
  |---|---|---|---|
  | ZS | ZS | nein | phi4-mini |
  | ZS+FT | ZS | ja | phase2-ft |
  | CoT+FS+AD | CoT+FS+AD | nein | phi4-mini |
  | CoT+FS+AD+FT | CoT+FS+AD | ja | phase2-ft |
- Metriken (Accuracy/AUROC/F1) werden von der bestehenden `metrics.py` automatisch in `phase2_results.xlsx` geschrieben.

## 5. Dateien (anlegen/ändern)

| Aktion | Datei | Zweck |
|---|---|---|
| Rewrite | `hateful_memes_app/finetune.py` | `build_finetune_data()` (ersetzt das alte deutsche Daten-Prep) |
| Neu | `hateful_memes_app/train_qlora.py` | Unsloth-QLoRA + GGUF-Export |
| Neu | `hateful_memes_app/launch_runpod_ft.sh` | RunPod-Setup fürs Training |
| Neu | `hateful_memes_app/tests/test_finetune.py` | Unit-Tests fürs Daten-Bauen |
| Ändern (E1) | `hateful_memes_app/phase2.py` | Prompt korrekt `.format()`-en, redundanten User-Block entfernen (RAG-Verhalten erhalten) |
| Neu | `hateful_memes_app/requirements-train.txt` | Trainings-Deps getrennt (App-`requirements.txt` bleibt schlank) |
| Ändern | `config*.json` | `ft_model_path`, dev-`phase1_excel`/Sheet |

## 6. Risiken & Caveats
- **Unsloth-GGUF-Konversion** (llama.cpp-Build) ist erfahrungsgemäß zickig → Zeitpuffer.
- **Basismodell-Parität:** Nicht-FT-Baseline nutzt Ollamas `phi4-mini` (Q4), FT nutzt unser Q4_K_M-GGUF desselben Phi-4-mini-instruct. Gleiche Basis, minimaler Quant-/Template-Confound — dokumentieren.
- **Chat-Template-Round-Trip** Phi-4 → GGUF → Ollama muss korrekt sein (Unsloth-Modelfile übernimmt Stop-Tokens/Template; nach `ollama create` an 1–2 Beispielen prüfen).
- **`max_seq_length`:** CoT+FS+AD-Prompt + lange Beschreibung muss < 4096 Tokens bleiben (vorab Längen prüfen).
- **8 GB M2** reicht für Inferenz eines 2.5-GB-Modells; Training läuft ausschließlich auf RunPod.

## 6b. Limitations (für Methoden-/Limitationsteil der Arbeit)

Aus dem Code-Review (2026-06-09). Diese Punkte sind bewusst akzeptierte bzw. zu benennende Einschränkungen:

- **L1 — FT-AUROC = Balanced Accuracy (konstruktionsbedingt).** Die Trainings-`confidence` wird aus dem Gold-Label in zwei nicht überlappende Bänder gesetzt (`label=1`→[82,96], `label=0`→[4,18]). Ein Modell, das das lernt, gibt eine Confidence aus, die faktisch nur das prädizierte Label wiederholt. Für solche Scores gilt `AUROC = (TPR+TNR)/2 = Balanced Accuracy`. **Konsequenz:** Die AUROC des FT-Modells trägt keine eigenständige Ranking-Information und ist **nicht** mit der AUROC der Nicht-FT-Modelle (selbstberichtete, gradierte `confidence`) vergleichbar. **Entscheidung:** FT-Bedingung wird auf **Accuracy/F1** verglichen; FT-AUROC wird nicht als eigenständige Metrik interpretiert. (Saubere Alternative für später: P(hateful) aus Label-Token-Logprobs statt verbalisierter Confidence — vereinheitlicht FT und Baseline, siehe §7.)
- **L2 — Selbstberichtete Confidence (Nicht-FT) ist schlecht kalibriert.** `confidence/100` als P(hateful) ist ein etablierter „verbalized confidence"-Proxy, aber: LLM-Confidence ist unkalibriert, das `"confidence": 75`-Beispiel in jedem Prompt ankert die Ausgaben, und grobe/diskrete Werte erzeugen viele Ties. Als Proxy okay, in Limitations zu benennen.
- **L3 — Ungleiche `n_samples` je Kombination.** `parse_error`/Timeout-Fälle werden aus den Metriken ausgeschlossen → jede Kombination evaluiert ggf. auf einer anderen (tendenziell leichteren) Teilmenge. `n_samples` wird mitgeführt; **Coverage (`n_samples/total`) ist beim Vergleich prominent auszuweisen.** (Alternative: `parse_error` als Fehlklassifikation werten.)
- **L4 — Asymmetrische Trainings-Reasonings.** Alle 837 hateful-Beispiele tragen reichhaltige menschliche HatReD-Begründungen, alle 1000 benign-Beispiele exakt **denselben** `BENIGN_TEMPLATE`. Das erzeugt ein Längen-/Spezifitäts-Signal, das mit dem Label korreliert, und die Benign-`reasoning` des Modells ist gelernte Boilerplate (für die negative Klasse kein echtes CoT). Pragmatisch vertretbar, als Limitation zu benennen.
- **L5 — Basismodell-/Template-Parität (zur Trainingszeit zu verifizieren).** Nicht-FT-Baseline = Ollamas `phi4-mini` (Q4), FT = unser Q4_K_M-GGUF. Vor dem Vergleich `ollama show phi4-mini --modelfile` gegen das exportierte Modelfile prüfen (gleiches TEMPLATE? gleiches Quant? injiziertes SYSTEM?). Bei Abweichung misst der Vergleich teils Quant/Template statt FT.
- **L6 — `train_on_responses_only`-Delimiter (zur Trainingszeit zu verifizieren).** `instruction_part`/`response_part` müssen exakt dem gerenderten Phi-4-mini-Chat-Template entsprechen, sonst wird der Loss falsch maskiert. Vor dem vollen Lauf an einem Beispiel die maskierten Labels prüfen.

## 7. Nicht im Scope (YAGNI)
- Fine-Tuning des Phase-1-VLM (multimodal) — nicht in der Matrix.
- RAG-Zellen mit FT (`+RAG+FT`) — separat, später.
- Prinzipielleres AUROC-Scoring per Label-Token-Logprob statt selbst gemeldeter `confidence` — als mögliche spätere Verbesserung notiert, ändert die bestehende Pipeline nicht.
- Hyperparameter-Sweeps — erst ein sauberer Durchlauf, dann ggf. optimieren.

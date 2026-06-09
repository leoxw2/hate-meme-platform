# RunPod Phase 1 Deployment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 der Hateful-Memes-Pipeline (Bildbeschreibung via qwen3-vl:4b) auf einer RunPod RTX-3090-VM ausführen, um 2000 Trainingsbilder zu beschriften.

**Architecture:** Streamlit-App + Ollama laufen auf der VM in getrennten tmux-Sessions. Ein separates `config_runpod.json` enthält Linux-Pfade, sodass die lokale Mac-Konfiguration unangetastet bleibt. Ergebnisse werden per rsync zurück auf den Mac geholt.

**Tech Stack:** RunPod (Ubuntu 22.04 + CUDA), Ollama, qwen3-vl:4b, Python 3.11, tmux, rsync

---

## Dateien-Überblick

| Aktion | Pfad | Zweck |
|--------|------|-------|
| Erstellen (lokal) | `hateful_memes_app/config_runpod.json` | Linux-Pfade für die VM |
| Erstellen (lokal) | `hateful_memes_app/launch_runpod.sh` | Ein-Klick-Startskript für die VM |
| Unverändert | `hateful_memes_app/config.py` | Liest config über `CONFIG_PATH`-Env-Variable |

---

## Task 1: RunPod-Pod erstellen

**Vorbedingung:** RunPod-Account, SSH-Key in RunPod hinterlegt (`https://www.runpod.io/console/user/settings` → SSH Keys).

- [ ] **Schritt 1: SSH-Key prüfen**

```bash
# Auf dem Mac: Public Key anzeigen
cat ~/.ssh/id_ed25519.pub
# Falls kein Key existiert:
ssh-keygen -t ed25519 -C "runpod-hatememe"
cat ~/.ssh/id_ed25519.pub
```

Diesen Public Key in RunPod unter Settings → SSH Keys einfügen.

- [ ] **Schritt 2: Pod auf RunPod.io erstellen**

1. Auf https://www.runpod.io/console/pods → **+ Deploy**
2. GPU: **RTX 3090** (24 GB VRAM, ausreichend für qwen3-vl:4b)
3. Template: **RunPod Pytorch 2.1** (enthält CUDA 12.1, Ubuntu 22.04)
4. Container Disk: **50 GB** (Modell ~5 GB + Bilder ~1 GB + Puffer)
5. **"Stop Idle Pod" → DEAKTIVIEREN** (sonst stoppt die VM bei SSH-Inaktivität)
6. Pod starten → SSH-Verbindungsdaten notieren (Host + Port)

- [ ] **Schritt 3: SSH-Verbindung testen**

```bash
# Beispiel-Befehl (Host/Port aus RunPod-Dashboard kopieren):
ssh -p <PORT> root@<HOST>
# Erwartete Ausgabe: root@<pod-id>:~#
exit
```

---

## Task 2: Lokale Deployment-Dateien erstellen

**Files:**
- Create: `hateful_memes_app/config_runpod.json`
- Create: `hateful_memes_app/launch_runpod.sh`

- [ ] **Schritt 1: `config_runpod.json` erstellen**

Dieses Config-File verwendet Linux-Pfade relativ zum Upload-Verzeichnis `/root/hatememe/`.

```json
{
  "prompt_excel": "/root/hatememe/data/prompts.xlsx",
  "img_folder": "/root/hatememe/data/img_train",
  "results_folder": "/root/hatememe/results",
  "max_tokens_phase1": 2500,
  "max_time_seconds": 120,
  "phase1_excel": "/root/hatememe/results/phase1_results.xlsx",
  "phase2_excel": "/root/hatememe/results/phase2_results.xlsx",
  "ft_model_path": ""
}
```

Speichern als `hateful_memes_app/config_runpod.json`.

- [ ] **Schritt 2: `launch_runpod.sh` erstellen**

Dieses Skript läuft einmalig auf der VM und richtet alles ein.

```bash
#!/usr/bin/env bash
set -e

echo "=== 1/5 System-Pakete ==="
apt-get update -qq && apt-get install -y -qq tmux curl rsync

echo "=== 2/5 Ollama installieren ==="
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== 3/5 Python-Abhängigkeiten ==="
cd /root/hatememe/hateful_memes_app
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "=== 4/5 Ergebnis-Ordner anlegen ==="
mkdir -p /root/hatememe/results

echo "=== 5/5 Ollama-Dienst + Modell-Download ==="
# Ollama im Hintergrund starten (dauert ein paar Sekunden)
tmux new-session -d -s ollama "ollama serve 2>&1 | tee /root/ollama.log"
sleep 5
ollama pull qwen3-vl:4b

echo ""
echo "Setup abgeschlossen. Starte jetzt Streamlit:"
echo "  tmux new-session -d -s app 'cd /root/hatememe/hateful_memes_app && source .venv/bin/activate && CONFIG_PATH=/root/hatememe/hateful_memes_app/config_runpod.json streamlit run app.py --server.port 8501 --server.headless true 2>&1 | tee /root/streamlit.log'"
```

Ausführbar machen:
```bash
chmod +x hateful_memes_app/launch_runpod.sh
```

---

## Task 3: Dateien auf die VM hochladen

**Vorbedingung:** Pod läuft, SSH-Verbindungsdaten bekannt. `<PORT>` und `<HOST>` aus RunPod-Dashboard ersetzen.

- [ ] **Schritt 1: Zielverzeichnisse auf VM anlegen**

```bash
ssh -p <PORT> root@<HOST> "mkdir -p /root/hatememe/data/img_train /root/hatememe/results"
```

Erwartete Ausgabe: kein Fehler.

- [ ] **Schritt 2: App-Code hochladen**

Aus dem lokalen Projektverzeichnis:
```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform

rsync -avz --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  -e "ssh -p <PORT>" \
  hateful_memes_app/ \
  root@<HOST>:/root/hatememe/hateful_memes_app/
```

Erwartete Ausgabe: Alle `.py`-Dateien + `config_runpod.json` + `launch_runpod.sh` werden übertragen.

- [ ] **Schritt 3: Prompts-Excel hochladen**

```bash
rsync -avz --progress \
  -e "ssh -p <PORT>" \
  data/prompts.xlsx \
  root@<HOST>:/root/hatememe/data/prompts.xlsx
```

- [ ] **Schritt 4: train.jsonl hochladen**

```bash
rsync -avz --progress \
  -e "ssh -p <PORT>" \
  data/train.jsonl \
  root@<HOST>:/root/hatememe/data/train.jsonl
```

- [ ] **Schritt 5: Trainingsbilder hochladen (2000 Bilder, ca. 1–3 GB)**

```bash
rsync -avz --progress \
  -e "ssh -p <PORT>" \
  data/img_train/ \
  root@<HOST>:/root/hatememe/data/img_train/
```

Erwartete Ausgabe: `2000 files transferred`. Dauert je nach Verbindung 5–20 Minuten.

- [ ] **Schritt 6: Upload verifizieren**

```bash
ssh -p <PORT> root@<HOST> "find /root/hatememe/data/img_train -type f | wc -l"
```

Erwartete Ausgabe: `2000`

---

## Task 4: VM-Umgebung einrichten

- [ ] **Schritt 1: Per SSH auf die VM einloggen**

```bash
ssh -p <PORT> root@<HOST>
```

- [ ] **Schritt 2: Setup-Skript ausführen**

```bash
bash /root/hatememe/hateful_memes_app/launch_runpod.sh
```

Erwartete Ausgabe: Alle 5 Schritte erscheinen ohne Fehler. Der Modell-Download (qwen3-vl:4b, ~2.5 GB) dauert 1–5 Minuten.

- [ ] **Schritt 3: Ollama-Dienst prüfen**

```bash
ollama list
```

Erwartete Ausgabe:
```
NAME              ID       SIZE    MODIFIED
qwen3-vl:4b       ...      2.5 GB  ...
```

---

## Task 5: Streamlit-App in tmux starten

- [ ] **Schritt 1: App-Session starten**

```bash
tmux new-session -d -s app \
  "cd /root/hatememe/hateful_memes_app && \
   source .venv/bin/activate && \
   CONFIG_PATH=/root/hatememe/hateful_memes_app/config_runpod.json \
   streamlit run app.py --server.port 8501 --server.headless true \
   2>&1 | tee /root/streamlit.log"
```

- [ ] **Schritt 2: Start verifizieren**

```bash
sleep 5 && tmux capture-pane -p -t app | tail -10
```

Erwartete Ausgabe:
```
You can now view your Streamlit app in your browser.
Network URL: http://0.0.0.0:8501
```

- [ ] **Schritt 3: Port-Forwarding auf dem Mac öffnen (neues Terminal-Fenster)**

```bash
ssh -p <PORT> -L 8501:localhost:8501 root@<HOST> -N
```

Danach im Browser auf dem Mac öffnen: **http://localhost:8501**

Die Streamlit-App sollte sichtbar sein.

---

## Task 6: Phase 1 im Fine-Tuning-Modus starten

Die App läuft im Browser. Die folgenden Schritte erfolgen über die Streamlit-UI.

- [ ] **Schritt 1: Modus wählen**

In der Sidebar:
- Modus: **Fine-Tuning-Modus (train.jsonl)**
- JSONL-Pfad: `/root/hatememe/data/train.jsonl`

- [ ] **Schritt 2: Konfiguration prüfen**

In der Sidebar unter Konfiguration:
- `img_folder`: `/root/hatememe/data/img_train`
- `results_folder`: `/root/hatememe/results`
- `phase1_excel`: `/root/hatememe/results/phase1_results.xlsx`

Falls Felder leer sind: Werte manuell eintragen und Konfiguration speichern.

- [ ] **Schritt 3: Prompt auswählen**

Einen vorhandenen Prompt aus `prompts.xlsx` auswählen (oder den ersten/Standard-Prompt nehmen).

- [ ] **Schritt 4: Phase 1 starten**

Auf **"Phase 1 starten"** klicken.

Erwartete Ausgabe: Fortschrittsbalken beginnt zu laufen, Logs erscheinen (Bild-ID + Beschreibung).

- [ ] **Schritt 5: Laufenden Prozess im SSH-Log beobachten**

In einem anderen Terminal (SSH auf VM):
```bash
tail -f /root/streamlit.log
```

- [ ] **Schritt 6: Checkpoint prüfen (nach ~100 Bildern)**

```bash
ls -lh /root/hatememe/results/
```

Erwartete Ausgabe: Eine Datei `phase1_<prompt_name>.csv` mit wachsender Größe.

---

## Task 7: Laufzeit einschätzen und SSH-Verbindung schließen

- [ ] **Schritt 1: Geschätzte Laufzeit berechnen**

Mit dem Modell qwen3-vl:4b auf RTX 3090 sind ~5–15 Sekunden pro Bild realistisch.
- 2000 Bilder × 10 Sek = ~5.5 Stunden (optimistisch)
- 2000 Bilder × 15 Sek = ~8 Stunden (konservativ)

- [ ] **Schritt 2: SSH-Verbindung sicher schließen**

Da App und Ollama in tmux laufen, können SSH-Verbindungen jederzeit getrennt werden — die Prozesse laufen weiter.

```bash
# Im SSH-Terminal:
exit
```

- [ ] **Schritt 3: Fortschritt später prüfen (erneut SSH)**

```bash
ssh -p <PORT> root@<HOST> "wc -l /root/hatememe/results/phase1_*.csv"
```

Erwartete Ausgabe: Zeilenzahl wächst gegen 2000 (+ 1 Header-Zeile = 2001).

---

## Task 8: Ergebnisse zurück auf den Mac laden

- [ ] **Schritt 1: Warten bis Phase 1 abgeschlossen**

```bash
ssh -p <PORT> root@<HOST> "tail -5 /root/streamlit.log"
```

Erwartete Ausgabe: `Phase 1 abgeschlossen` oder `done` im Log.

- [ ] **Schritt 2: Ergebnisse per rsync herunterladen**

Auf dem Mac:
```bash
cd /Users/leo/Claude_Projekte/hate_meme_platform

rsync -avz --progress \
  -e "ssh -p <PORT>" \
  root@<HOST>:/root/hatememe/results/ \
  data/results_runpod/
```

Erwartete Ausgabe: CSV-Checkpoint-Datei(en) + `phase1_results.xlsx` werden heruntergeladen.

- [ ] **Schritt 3: Daten verifizieren**

```bash
wc -l data/results_runpod/phase1_*.csv
```

Erwartete Ausgabe: `2001` (2000 Bilder + 1 Header-Zeile).

- [ ] **Schritt 4: Pod stoppen**

Im RunPod-Dashboard: Pod auswählen → **Stop Pod** klicken.

> **Achtung:** Nicht „Terminate" klicken — das löscht alle Daten auf dem Pod unwiderruflich. „Stop" reicht, um die Abrechnung zu beenden.

---

## Abbruch und Fortsetzung

Falls die VM neu gestartet werden muss oder ein Fehler auftritt:

```bash
# VM neu starten
ssh -p <PORT> root@<HOST>

# Ollama wieder starten (falls nicht mehr läuft)
tmux new-session -d -s ollama "ollama serve 2>&1 | tee /root/ollama.log"
sleep 5

# App neu starten
tmux new-session -d -s app \
  "cd /root/hatememe/hateful_memes_app && \
   source .venv/bin/activate && \
   CONFIG_PATH=/root/hatememe/hateful_memes_app/config_runpod.json \
   streamlit run app.py --server.port 8501 --server.headless true \
   2>&1 | tee /root/streamlit.log"
```

In der Streamlit-UI: Phase 1 erneut starten. Durch den CSV-Checkpoint werden bereits verarbeitete Bilder **automatisch übersprungen**.

---

## Kostenschätzung

| GPU | Preis (ca.) | 8h Laufzeit |
|-----|------------|-------------|
| RTX 3090 | ~$0.44/h | ~$3.50 |
| RTX 4090 | ~$0.79/h | ~$6.30 (dafür ggf. schneller) |

---

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `ollama: command not found` | `tmux attach -t ollama` prüfen, ggf. neu installieren |
| Streamlit startet nicht | `tmux attach -t app` → Fehlermeldung lesen |
| Bilder fehlen (`missing_image`) | Pfad `img_folder` in config_runpod.json prüfen, `ls /root/hatememe/data/img_train/` |
| SSH-Verbindung bricht ab | Kein Problem, tmux-Sessions laufen weiter |
| Pod wurde gestoppt (Idle) | `Stop Idle Pod` war aktiviert → Pod neu starten, CSV-Checkpoint bleibt erhalten |

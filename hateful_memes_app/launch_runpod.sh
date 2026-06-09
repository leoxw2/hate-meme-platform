#!/usr/bin/env bash
set -e

echo "=== 1/5 System-Pakete ==="
apt-get update -qq && apt-get install -y -qq tmux curl rsync

echo "=== 2/5 Ollama installieren ==="
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== 3/5 Python-Abhaengigkeiten ==="
cd /root/hatememe/hateful_memes_app
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "=== 4/5 Ergebnis-Ordner anlegen ==="
mkdir -p /root/hatememe/results

echo "=== 5/5 Ollama-Dienst + Modell-Download ==="
tmux new-session -d -s ollama "ollama serve 2>&1 | tee /root/ollama.log"
sleep 5
ollama pull qwen3-vl:4b

echo ""
echo "Setup fertig! Streamlit starten mit:"
echo "  tmux new-session -d -s app 'cd /root/hatememe/hateful_memes_app && source .venv/bin/activate && CONFIG_PATH=/root/hatememe/hateful_memes_app/config_runpod.json streamlit run app.py --server.port 8501 --server.headless true 2>&1 | tee /root/streamlit.log'"

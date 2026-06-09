#!/usr/bin/env bash
# hateful_memes_app/launch_runpod_ft.sh
#
# Run on the RunPod VM from /root/hatememe/:
#   bash hateful_memes_app/launch_runpod_ft.sh
#
# Prerequisites:
#   - Files uploaded to /root/hatememe/ via rsync (see plan Task 7)
#   - data/finetune_data.jsonl present (2000 lines)
set -e

echo "=== 1/5 System packages ==="
apt-get update -qq && apt-get install -y -qq tmux rsync

echo "=== 2/5 Verify GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "=== 3/5 Install training dependencies ==="
pip install --upgrade pip -q
# Install unsloth first (handles its own torch/xformers deps), then the rest without deps
pip install "unsloth" -q
pip install --no-deps trl peft accelerate bitsandbytes datasets -q

echo "=== 4/5 Verify Unsloth import ==="
python3 -c "from unsloth import FastLanguageModel; print('unsloth OK')"

echo "=== 5/5 Launch training in tmux session 'ft' ==="
tmux new-session -d -s ft \
  "cd /root/hatememe && python hateful_memes_app/train_qlora.py 2>&1 | tee /root/train.log"

echo ""
echo "Training started in tmux session 'ft'."
echo "Monitor with:  tmux attach -t ft"
echo "Or tail log:   tail -f /root/train.log"
echo ""
echo "Expected runtime on RTX 3090: ~45-90 min"

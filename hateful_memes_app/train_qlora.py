# hateful_memes_app/train_qlora.py
"""
QLoRA fine-tuning of Phi-4-mini-instruct via Unsloth.
Run on RunPod (RTX 3090, 24 GB VRAM).

Usage (from /root/hatememe/):
    python hateful_memes_app/train_qlora.py

Outputs:
    phase2-ft-gguf/          <- GGUF file (q4_k_m) + Modelfile
"""
import json, os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Phi-4-mini-instruct"
DATA_PATH      = "data/finetune_data.jsonl"
OUTPUT_DIR     = "phase2-ft-gguf"
MAX_SEQ_LENGTH = 4096
SEED           = 42
EPOCHS         = 2
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 2e-4
WARMUP_RATIO   = 0.03

# ── Imports (only available on RunPod after: pip install -r requirements-train.txt) ──
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer


def load_dataset(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} examples from {path}")
    return Dataset.from_list(rows)


def main():
    # 1. Load base model in 4-bit
    print("Loading model …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,          # auto-detect (bf16 on Ampere)
    )

    # 2. Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # 3. Load and format data
    dataset = load_dataset(DATA_PATH)

    def apply_chat_template(examples):
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True,
                          remove_columns=["messages"])

    # 4. Sanity check: print first formatted example (truncated)
    print("\n── First training example (first 600 chars) ──")
    print(dataset[0]["text"][:600])
    print("…\n")

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            bf16=True,
            optim="adamw_8bit",
            seed=SEED,
            output_dir="checkpoints",
            logging_steps=25,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    # 6. Train only on assistant responses
    # Phi-4-mini chat template uses <|user|> / <|assistant|> delimiters.
    # Verify with: print(tokenizer.apply_chat_template(
    #     [{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}],
    #     tokenize=False)) — look for the exact strings around the turns.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|user|>\n",
        response_part="<|assistant|>\n",
    )

    # 7. Train
    print("Starting training …")
    trainer.train()
    print("Training complete.")

    # 8. Export GGUF (q4_k_m) — creates OUTPUT_DIR/*.gguf + Modelfile
    print(f"Exporting GGUF → {OUTPUT_DIR}/ …")
    model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")

    print("\n── Export complete ──")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print(f"  {fname}  ({size / 1e9:.2f} GB)")

    print(f"""
Next steps on Mac:
  rsync -avz -e 'ssh -p <PORT>' root@<HOST>:/root/hatememe/{OUTPUT_DIR}/ ./{OUTPUT_DIR}/
  ollama create phase2-ft -f {OUTPUT_DIR}/Modelfile
  # then in config.json: "ft_model_path": "phase2-ft"
""")


if __name__ == "__main__":
    main()

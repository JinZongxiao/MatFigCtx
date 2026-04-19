"""
Continued Pre-training (CPT) for domain knowledge injection.

Injects materials science knowledge from MatFigCtx-generated SIS into
Gemma 4 E2B via LoRA CPT in chat format.

Data source: sis_outputs/all_training_data.json
  (produced by App 1 / pipeline.py --app 1)
  Each record: {"image": ..., "caption": ..., "sis": "...", "source": ...}

Usage:
    python cpt_gemma.py \
        --sis-json /path/to/sis_outputs/all_training_data.json \
        --model    /path/to/gemma-4-E2B-it \
        --output   /path/to/cpt_checkpoints
"""

import argparse
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Defaults — override via CLI or config
MODEL_PATH   = "/path/to/gemma-4-E2B-it"
SIS_JSON     = "/path/to/sis_outputs/all_training_data.json"
OUTPUT_DIR   = "/path/to/cpt_checkpoints"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
BATCH_SIZE   = 2
GRAD_ACCUM   = 8
LR           = 2e-5
EPOCHS       = 5
MAX_LEN      = 512

USER_PROMPT = "You are a materials science expert. Summarize the key scientific observations from this characterization result."


class CPTDataset(Dataset):
    def __init__(self, sis_json: str, tokenizer, max_len=MAX_LEN,
                 held_out_ratio: float = 0.1, seed: int = 42):
        """
        Load SIS from all_training_data.json (output of App 1).

        The last `held_out_ratio` fraction (by the same seed as eval_ppl.py)
        is excluded from training so that eval_ppl.py gets a clean held-out set.
        """
        import random
        self.tokenizer = tokenizer
        self.max_len   = max_len

        with open(sis_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        valid = [
            str(rec.get("sis", "")).strip() for rec in records
            if str(rec.get("sis", "")).strip()
            and not str(rec.get("sis", "")).startswith("[ERROR]")
            and len(str(rec.get("sis", "")).strip()) >= 20
        ]

        # Deterministic shuffle — same split as eval_ppl.py
        rng = random.Random(seed)
        rng.shuffle(valid)
        n_held = max(1, int(len(valid) * held_out_ratio))
        self.samples = valid[:-n_held]     # training split (first 90%)

        print(f"[INFO] CPT dataset: {len(self.samples)} train / "
              f"{n_held} held-out  (seed={seed})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _to_list(ids):
        """Normalise apply_chat_template output to a flat Python list."""
        if hasattr(ids, "ids"):             # tokenizers.Encoding
            return ids.ids
        if isinstance(ids, dict):           # BatchEncoding
            ids = ids["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]                    # [[...]] → [...]
        return ids

    def __getitem__(self, idx):
        text = self.samples[idx]

        messages = [
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": text},
        ]
        user_msgs = [{"role": "user", "content": USER_PROMPT}]

        full_list = self._to_list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False))
        user_list = self._to_list(self.tokenizer.apply_chat_template(
            user_msgs, tokenize=True, add_generation_prompt=True))

        user_len  = len(user_list)
        input_ids = torch.tensor(full_list[:self.max_len], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        labels = input_ids.clone()
        labels[:min(user_len, labels.shape[0])] = -100  # mask user部分
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def replace_clippable_linear(m):
    for name, child in m.named_children():
        if "Clippable" in type(child).__name__:
            setattr(m, name, child.linear)
        else:
            replace_clippable_linear(child)


def main(args=None):
    model_path = args.model if args else MODEL_PATH
    output_dir = args.output if args else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    replace_clippable_linear(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    for name, param in model.named_parameters():
        if "vision_tower" in name or "audio_tower" in name:
            param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "per_layer_input_gate", "per_layer_projection"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = CPTDataset(args.sis_json, tokenizer)

    from torch.nn.utils.rnn import pad_sequence
    def collate_fn(batch):
        return {
            "input_ids":      pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
            "labels":         pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print("[INFO] Start CPT...")
    trainer.train()

    lora_path = os.path.join(output_dir, "lora_cpt_final")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"[INFO] CPT LoRA saved to {lora_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPT knowledge injection for Gemma 4 E2B")
    parser.add_argument("--sis-json", default=SIS_JSON,
                        help="Path to all_training_data.json (App 1 output)")
    parser.add_argument("--model",    default=MODEL_PATH, help="Gemma model path")
    parser.add_argument("--output",   default=OUTPUT_DIR, help="Output checkpoint directory")
    args = parser.parse_args()
    main(args)
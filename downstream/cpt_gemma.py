import argparse
import csv
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

STATEMENT_PROMPT = "State scientific findings reported in the materials-science literature."
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "per_layer_input_gate", "per_layer_projection",
]


def unwrap_clippable_linear(module):
    # PEFT doesn't wrap Gemma 4's ClippableLinear directly
    for name, child in module.named_children():
        if "Clippable" in type(child).__name__:
            setattr(module, name, child.linear)
        else:
            unwrap_clippable_linear(child)


class PairDataset(Dataset):
    def __init__(self, path: Path, tok, max_len: int = 512):
        self.rows = json.loads(path.read_text(encoding="utf-8"))
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        user = self.tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )["input_ids"][0]
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )["input_ids"][0][: self.max_len]
        labels = full.clone()
        labels[: min(len(user), len(labels))] = -100
        return {"input_ids": full, "labels": labels}


def train_lora(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    unwrap_clippable_linear(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # freeze multimodal towers, adapt text LoRA only
    for name, p in model.named_parameters():
        if "vision_tower" in name or "audio_tower" in name:
            p.requires_grad = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    data = PairDataset(args.train_json, tok, max_len=args.max_len)

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for k, b in enumerate(batch):
            m = len(b["input_ids"])
            ids[k, :m], lab[k, :m], att[k, :m] = b["input_ids"], b["labels"], 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )
    Trainer(model=model, args=targs, train_dataset=data, data_collator=collate).train()
    final_dir = args.out_dir / "final"
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print(f"Saved adapter to {final_dir}")


@torch.no_grad()
def eval_nll(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    if args.lora_dir:
        unwrap_clippable_linear(model)
        model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()

    conclusions = json.loads(args.conclusions_json.read_text(encoding="utf-8"))
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
    rows = []
    for doc, text in sorted(conclusions.items()):
        text = (text or "").strip()
        if len(text) < 200:
            continue
        user = tok.apply_chat_template([{"role": "user", "content": STATEMENT_PROMPT}], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(
            [{"role": "user", "content": STATEMENT_PROMPT}, {"role": "assistant", "content": text}],
            tokenize=False, add_generation_prompt=False,
        )
        ids = tok.encode(full, add_special_tokens=False)[: args.max_len]
        n_user = len(tok.encode(user, add_special_tokens=False))
        seq = torch.tensor(ids, device=model.device)
        labels = seq.clone()
        labels[:n_user] = -100
        logits = model(seq.unsqueeze(0)).logits[0].float()
        rows.append({"doc": doc, "nll": loss_fn(logits[:-1], labels[1:]).item(), "tokens": int((labels[1:] != -100).sum())})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "nll", "tokens"])
        w.writeheader()
        w.writerows(rows)
    mean_nll = sum(r["nll"] for r in rows) / max(1, sum(r["tokens"] for r in rows))
    print(f"Evaluated {len(rows)} conclusions: mean NLL = {mean_nll:.4f} nats/token")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--model", required=True)
    tr.add_argument("--train-json", type=Path, required=True)
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--lora-rank", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=32)
    tr.add_argument("--lora-dropout", type=float, default=0.05)
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--grad-accum", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--epochs", type=float, default=1.0)
    tr.add_argument("--max-len", type=int, default=512)
    tr.add_argument("--seed", type=int, default=0)

    ev = sub.add_parser("eval-nll")
    ev.add_argument("--model", required=True)
    ev.add_argument("--lora-dir", type=str, default=None)
    ev.add_argument("--conclusions-json", type=Path, required=True)
    ev.add_argument("--out-csv", type=Path, required=True)
    ev.add_argument("--max-len", type=int, default=512)

    args = ap.parse_args()
    if args.cmd == "train":
        train_lora(args)
    else:
        eval_nll(args)


if __name__ == "__main__":
    main()

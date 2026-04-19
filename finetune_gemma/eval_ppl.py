"""
eval_ppl.py — Perplexity evaluation for domain knowledge injection.

Compares perplexity on held-out ODS-W SIS between:
  - Base Gemma 4 E2B (no fine-tuning)
  - CPT Gemma 4 E2B (after LoRA knowledge injection)

Lower PPL = model is more "familiar" with materials science SIS text
         = successful domain knowledge injection.

Usage (run AFTER cpt_gemma.py finishes):

    # Compare base vs. CPT
    python eval_ppl.py \\
        --sis-json   ./all_training_data.json \\
        --model      /path/to/gemma-4-E2B-it \\
        --lora-path  ./cpt_checkpoints/lora_cpt_final \\
        --held-out-ratio 0.1 \\
        --seed 42

    # Base model only (sanity check before CPT finishes)
    python eval_ppl.py \\
        --sis-json ./all_training_data.json \\
        --model    /path/to/gemma-4-E2B-it \\
        --base-only
"""

import argparse
import json
import math
import random

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
MAX_LEN     = 512
BATCH_SIZE  = 1          # keep at 1 for variable-length sequences
USER_PROMPT = (
    "You are a materials science expert. "
    "Summarize the key scientific observations from this characterization result."
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def load_held_out(sis_json: str, held_out_ratio: float, seed: int):
    """Return held-out SIS strings (deterministic split by seed)."""
    with open(sis_json, encoding="utf-8") as f:
        data = json.load(f)

    valid = [
        r["sis"] for r in data
        if str(r.get("sis", "")).strip()
        and not str(r.get("sis", "")).startswith("[ERROR]")
        and len(str(r.get("sis", "")).strip()) >= 20
    ]

    rng = random.Random(seed)
    rng.shuffle(valid)
    n_held = max(1, int(len(valid) * held_out_ratio))
    held_out = valid[-n_held:]          # last n_held are held-out (same split every time)
    train_n  = len(valid) - n_held

    print(f"[INFO] Total valid SIS : {len(valid)}")
    print(f"[INFO] Training split  : {train_n}  (first {100*(1-held_out_ratio):.0f}%)")
    print(f"[INFO] Held-out split  : {n_held}  (last {100*held_out_ratio:.0f}%)")
    return held_out


def replace_clippable_linear(m):
    """Fix Gemma 4 E2B compatibility issue (same as cpt_gemma.py)."""
    for name, child in m.named_children():
        if "Clippable" in type(child).__name__:
            setattr(m, name, child.linear)
        else:
            replace_clippable_linear(child)


def load_model(model_path: str, lora_path: str = None, device: str = "cuda"):
    """Load base model, optionally merge LoRA weights."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    replace_clippable_linear(model)

    if lora_path:
        print(f"[INFO] Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()   # merge LoRA into base weights
        print("[INFO] LoRA merged.")

    model.eval()
    return tokenizer, model


@torch.no_grad()
def compute_ppl(model, tokenizer, sis_list: list, device: str = "cuda") -> dict:
    """
    Compute perplexity on a list of SIS strings.

    Each SIS is formatted as an assistant turn in chat format.
    Only the assistant tokens (SIS) contribute to the loss;
    the user prompt tokens are masked with -100.

    Returns:
        {
          "ppl"        : float,   # overall perplexity
          "mean_loss"  : float,   # mean per-token NLL
          "n_tokens"   : int,     # total assistant tokens evaluated
          "n_samples"  : int,
        }
    """
    loss_fn   = CrossEntropyLoss(reduction="sum", ignore_index=-100)
    total_nll = 0.0
    total_tok = 0
    n_skip    = 0

    for sis in tqdm(sis_list, desc="Evaluating PPL"):
        # ── Build chat-format sequence ────────────────────
        messages_full = [
            {"role": "user",      "content": USER_PROMPT},
            {"role": "assistant", "content": sis},
        ]
        messages_user = [
            {"role": "user", "content": USER_PROMPT},
        ]

        full_text = tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        user_text = tokenizer.apply_chat_template(
            messages_user,
            tokenize=False,
            add_generation_prompt=True,
        )

        # tokenize=False always returns str → encode() always returns List[int]
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        user_ids = tokenizer.encode(user_text, add_special_tokens=False)

        seq      = torch.tensor(full_ids[:MAX_LEN], dtype=torch.long).to(device)
        user_len = len(user_ids)

        labels = seq.clone()
        labels[:user_len] = -100                   # mask user prompt
        labels[seq == tokenizer.pad_token_id] = -100

        # Count actual assistant tokens being evaluated
        n_assistant = (labels != -100).sum().item()
        if n_assistant == 0:
            n_skip += 1
            continue

        # ── Forward pass ─────────────────────────────────
        logits = model(seq.unsqueeze(0)).logits[0]  # [T, V]

        # Shift: predict token t+1 from position t
        shift_logits = logits[:-1]                  # [T-1, V]
        shift_labels = labels[1:]                   # [T-1]

        nll        = loss_fn(shift_logits, shift_labels).item()
        n_tok      = (shift_labels != -100).sum().item()
        total_nll += nll
        total_tok += n_tok

    if n_skip:
        print(f"[WARN] Skipped {n_skip} samples (no assistant tokens after truncation).")

    if total_tok == 0:
        return {"ppl": float("inf"), "mean_loss": float("inf"),
                "n_tokens": 0, "n_samples": len(sis_list)}

    mean_loss = total_nll / total_tok
    ppl       = math.exp(mean_loss)
    return {
        "ppl":       round(ppl, 4),
        "mean_loss": round(mean_loss, 6),
        "n_tokens":  total_tok,
        "n_samples": len(sis_list) - n_skip,
    }


def print_result(label: str, result: dict):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Perplexity (PPL) : {result['ppl']:.4f}")
    print(f"  Mean NLL / token : {result['mean_loss']:.6f}")
    print(f"  Tokens evaluated : {result['n_tokens']:,}")
    print(f"  Samples          : {result['n_samples']}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation for MatFigCtx knowledge injection"
    )
    parser.add_argument("--sis-json",        required=True,
                        help="Path to all_training_data.json")
    parser.add_argument("--model",           required=True,
                        help="Base Gemma model path")
    parser.add_argument("--lora-path",       default=None,
                        help="CPT LoRA checkpoint directory (omit for base-only)")
    parser.add_argument("--held-out-ratio",  type=float, default=0.1,
                        help="Fraction of SIS to hold out for evaluation (default: 0.1)")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--base-only",       action="store_true",
                        help="Only evaluate the base model (skip CPT comparison)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # ── Load held-out data ───────────────────────────────
    held_out = load_held_out(args.sis_json, args.held_out_ratio, args.seed)

    # ── Base model PPL ───────────────────────────────────
    print("\n[Step 1] Evaluating BASE Gemma...")
    tokenizer, base_model = load_model(args.model, lora_path=None, device=device)
    base_result = compute_ppl(base_model, tokenizer, held_out, device)
    print_result("Base Gemma (no fine-tuning)", base_result)

    del base_model
    torch.cuda.empty_cache()

    if args.base_only or args.lora_path is None:
        print("\n[Done] Base-only evaluation complete.")
        return

    # ── CPT model PPL ────────────────────────────────────
    print("\n[Step 2] Evaluating CPT Gemma (with LoRA)...")
    tokenizer, cpt_model = load_model(args.model, lora_path=args.lora_path, device=device)
    cpt_result = compute_ppl(cpt_model, tokenizer, held_out, device)
    print_result("CPT Gemma (after knowledge injection)", cpt_result)

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  Knowledge Injection Summary")
    print(f"{'═'*50}")
    print(f"  Base PPL : {base_result['ppl']:.4f}")
    print(f"  CPT  PPL : {cpt_result['ppl']:.4f}")
    delta = base_result["ppl"] - cpt_result["ppl"]
    pct   = delta / base_result["ppl"] * 100
    arrow = "↓" if delta > 0 else "↑"
    print(f"  Change   : {arrow} {abs(delta):.4f}  ({abs(pct):.1f}%)")
    if delta > 0:
        print("  → PPL decreased: domain knowledge successfully injected.")
    else:
        print("  → PPL did not decrease: check training loss and data quality.")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()

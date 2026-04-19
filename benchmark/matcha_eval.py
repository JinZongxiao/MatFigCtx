"""
benchmark/matcha_eval.py
MatCha benchmark 统一VQA评测，支持 zero_shot / this_work 两种 setting。
this_work setting 会从 matcha_pipeline.py 产出的 plugin CSV 读取参考证据。

Usage:
    python -m benchmark.matcha_eval --model qwen3-2b --setting zero_shot
    python -m benchmark.matcha_eval --model internvl-8b --setting this_work
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config
from benchmark._vlm_utils import load_vlm_model


SYSTEM_PROMPT = (
    "You are a helpful materials science assistant. "
    "Based on the figure, answer the following question. "
    "The answer must be concise and clear. "
    "Answer directly without any explanation."
)


def build_prompt(question, options, reference=None):
    opts_str = ""
    if options:
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                pass
        if isinstance(options, list):
            opts_str = "\n".join(options)
        elif isinstance(options, dict):
            opts_str = "\n".join(f"({k}) {v}" for k, v in options.items())
        else:
            opts_str = str(options)

    parts = []
    if reference:
        parts.append(f"Reference:\n{reference}")
    if opts_str:
        parts.append(f"Options:\n{opts_str}")
    parts.append(f"Question: {question}")
    parts.append("Answer with the option's letter from the given choices directly:")
    return "\n\n".join(parts)


def _run_qwen3(model, processor, img_path, prompt, max_tokens):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(img_path)},
            {"type": "text", "text": prompt},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def _run_internvl(model, processor, img_path, prompt, max_tokens):
    pil_image = Image.open(img_path).convert("RGB")
    hf_messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{prompt}"},
        ],
    }]
    text = processor.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[pil_image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def run_inference(model_key, model, processor, img_path, prompt, max_tokens=64):
    if model_key.startswith("qwen"):
        return _run_qwen3(model, processor, img_path, prompt, max_tokens)
    return _run_internvl(model, processor, img_path, prompt, max_tokens)


def main():
    parser = argparse.ArgumentParser(description="MatCha Benchmark Evaluation")
    parser.add_argument("--model",   type=str, required=True)
    parser.add_argument("--setting", type=str, required=True, choices=["zero_shot", "this_work"])
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = cfg["VLM_REGISTRY"]
    if args.model not in registry:
        raise ValueError(f"--model must be one of {list(registry)}")

    jsonl_path = cfg["MATCHA_JSONL"]
    img_root   = Path(cfg["MATCHA_IMG_ROOT"])
    output_dir = Path(cfg["MATCHA_OUTPUT_DIR"])
    plugin_dir = Path(cfg["MATCHA_PLUGIN_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{args.model}_{args.setting}.csv"
    if out_csv.exists():
        print(f"[SKIP] Already exists: {out_csv}")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f]

    plugin_df = None
    if args.setting == "this_work":
        plugin_file = "matcha_qwen38b.csv" if args.model.startswith("qwen") else "matcha_internvl8b.csv"
        plugin_path = plugin_dir / plugin_file
        if not plugin_path.exists():
            raise FileNotFoundError(
                f"Plugin CSV not found: {plugin_path}\n"
                f"Run `python -m benchmark.matcha_pipeline --model {'qwen3-8b' if args.model.startswith('qwen') else 'internvl-8b'}` first."
            )
        plugin_df = pd.read_csv(plugin_path)
        print(f"[INFO] Loaded plugin: {plugin_path} ({len(plugin_df)} rows)")

    model, processor = load_vlm_model(args.model, registry[args.model])

    rows = []
    for i, data in enumerate(tqdm(json_data)):
        vqa       = (data.get("vqa") or [{}])[0]
        question  = vqa.get("question", "")
        gt_answer = vqa.get("answer", "")
        options   = vqa.get("options", None)
        topic     = vqa.get("topic", "")
        img_rel   = (data.get("images") or [{}])[0].get("image_path", "")
        img_path  = img_root / img_rel

        reference = None
        if args.setting == "this_work" and plugin_df is not None:
            article_info = data.get("article_info", {})
            article_url = article_info.get("article_url", "") if isinstance(article_info, dict) else str(article_info)
            article_id = article_url.rsplit("/", 1)[-1] if article_url else ""
            matched = plugin_df[plugin_df["input_json"].str.contains(article_id, na=False)]
            if not matched.empty:
                reference = str(matched.iloc[0]["local_outputs"])

        row = {
            "idx": i, "topic": topic, "question": question, "gt_answer": gt_answer,
            "options": json.dumps(options, ensure_ascii=False) if options else "",
            "image_path": str(img_path), "model": args.model, "setting": args.setting,
            "pred": "", "error": "",
        }
        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            prompt = build_prompt(question, options, reference)
            row["pred"] = run_inference(args.model, model, processor, img_path, prompt)
        except Exception as e:
            row["error"] = repr(e)
            print(f"[ERROR] idx={i}: {e}")

        rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(json_data):
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_csv}")


if __name__ == "__main__":
    main()

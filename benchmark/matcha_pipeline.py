"""
benchmark/matcha_pipeline.py
在MatCha数据集上跑evidence chain，产出 matcha_eval.py 使用的 plugin CSV。

Usage:
    python -m benchmark.matcha_pipeline --model qwen3-8b
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config
from benchmark._vlm_utils import load_vlm_model, run_vlm, resize_image
from benchmark.ablation import build_seg_description, deduplicate


def build_prompt(img_caption, img_cap_text, n, seg_description):
    return (
        "You are an expert assistant in materials science literature understanding.\n"
        "Based on the following evidence, generate concise scientific statements "
        "describing the figure content.\n\n"
        f"[Figure Caption]:\n{img_caption}\n\n"
        f"[Body Text] (paragraphs directly referencing this figure):\n{img_cap_text}\n\n"
        f"[Subfigure Structure]: This figure contains {n} subfigure(s). {seg_description}\n\n"
        "Output requirements:\n"
        "- Each statement must be an independent scientific fact sentence (subject + verb + object).\n"
        "- Prioritize information from the figure caption and body text.\n"
        "- Be concise and precise, suitable for downstream NER tasks.\n"
        "- Do NOT add numbering, explanations, prefixes, or suffixes.\n"
        "- Do NOT introduce content not present in the provided text.\n"
        "- Output scientific statements ONLY, one per line."
    )


def main():
    parser = argparse.ArgumentParser(description="MatCha Evidence Chain Pipeline")
    parser.add_argument("--model",  type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry   = cfg["VLM_REGISTRY"]
    input_dir  = cfg["MATCHA_EVIDENCE_DIR"]
    plugin_dir = Path(cfg["MATCHA_PLUGIN_DIR"])
    cache_root = os.path.join(plugin_dir, "resized_images")
    image_size = int(cfg.get("IMAGE_SIZE", 512))
    max_tokens = int(cfg.get("MAX_NEW_TOKENS", 1024))

    if args.model not in registry:
        raise ValueError(f"--model must be one of {list(registry)}")

    plugin_dir.mkdir(parents=True, exist_ok=True)
    out_csv = plugin_dir / f"matcha_{args.model.replace('-', '')}.csv"
    if out_csv.exists():
        print(f"[SKIP] Already exists: {out_csv}")
        return

    model, processor = load_vlm_model(args.model, registry[args.model])

    outputs, error_rows = [], []
    json_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))

    for fname in tqdm(json_files):
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            error_rows.append({"input_json": fname, "img_index": -1, "error": str(e)})
            continue

        one_outputs = []
        for idx, item in enumerate(data):
            try:
                caption  = item.get("img_caption", "")
                cap_text = item.get("img_cap_text", "")
                sub_types = item.get("sub_image_type", []) or []
                sub_images = item.get("sub_images", []) or []
                img_path = item.get("img_path", "")
                n = len(sub_types)

                seg_desc = build_seg_description(sub_types, n)
                prompt = build_prompt(caption, cap_text, n, seg_desc).replace("\x00", "")

                all_imgs = [img_path] + list(sub_images)
                img_array = []
                for p in all_imgs:
                    if p and os.path.exists(p):
                        img_array.append(resize_image(p, image_size, cache_root))
                if not img_array:
                    raise FileNotFoundError(f"No valid images for idx={idx}")

                raw = run_vlm(args.model, model, processor, img_array, prompt, max_tokens)
                one_outputs.append(deduplicate(raw))
            except Exception as e:
                err = str(e)
                print(f"[ERROR] {fname} idx={idx}: {err}")
                one_outputs.append(f"[ERROR] {err}")
                error_rows.append({"input_json": fname, "img_index": idx, "error": err})

        outputs.append({"input_json": fname, "local_outputs": one_outputs})

    pd.DataFrame(outputs).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_csv}")
    if error_rows:
        err_path = str(out_csv).replace(".csv", "_errors.csv")
        pd.DataFrame(error_rows).to_csv(err_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

"""
benchmark/ablation.py
消融实验主脚本，支持 Qwen3-VL 与 InternVL3.5 系列、7种实验设置。

Usage:
    python -m benchmark.ablation --model qwen3-4b --setting full
    python -m benchmark.ablation --model internvl-8b --setting wo_local

Settings:
    zero_shot       只有图，无任何文本
    img_only        只有图（与zero_shot等价，用于对照）
    txt_only        只有文本，无图
    full            完整框架（图 + subfig_type + local + global）
    wo_subfig_type  去掉子图类型证据
    wo_local        去掉局部文本证据
    wo_global       去掉全局文档证据
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


def get_info(file_path):
    keys = ["img_path", "img_caption", "img_cap_text",
            "literature_title", "abstract", "conclusion",
            "sub_images", "sub_image_type"]
    arrays = {k: [] for k in keys}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for k in keys:
            arrays[k].append(item.get(k))
    return tuple(arrays[k] for k in keys)


def build_seg_description(sub_image_type, n):
    parts = []
    for i in range(n):
        t = sub_image_type[i].get("predicted_type", sub_image_type[i].get("type", "unknown"))
        prob = sub_image_type[i].get("predicted_prob", sub_image_type[i].get("prob", 0.0))
        parts.append(f"subfigure {i+1}: {t} (confidence {prob:.4f})")
    return "; ".join(parts)


def build_prompt(setting, global_message, img_caption, img_cap_text, n, seg_description):
    use_subfig = setting not in ("zero_shot", "img_only", "txt_only", "wo_subfig_type")
    use_local  = setting not in ("zero_shot", "img_only", "wo_local")
    use_global = setting not in ("zero_shot", "img_only", "wo_global")

    sections = [
        "You are an expert assistant in materials science literature understanding.\n"
        "Based on the following evidence, generate concise scientific statements "
        "describing the figure content."
    ]
    if use_global:
        sections.append(f"[Global Context] (from title, abstract, and conclusion):\n{global_message}")
    if use_local:
        sections.append(
            f"[Figure Caption]:\n{img_caption}\n\n"
            f"[Body Text] (paragraphs directly referencing this figure):\n{img_cap_text}"
        )
    if use_subfig:
        sections.append(f"[Subfigure Structure]: This figure contains {n} subfigure(s). {seg_description}")
    if setting == "txt_only":
        sections.append("(No image is provided. Generate statements based on text evidence only.)")
    sections.append(
        "Output requirements:\n"
        "- Each statement must be an independent scientific fact sentence (subject + verb + object).\n"
        "- Be concise and precise, suitable for downstream NER tasks.\n"
        "- Do NOT add numbering, explanations, prefixes, or suffixes.\n"
        "- Do NOT introduce content not present in the provided text.\n"
        "- Output scientific statements ONLY, one per line."
    )
    return "\n\n".join(sections)


def deduplicate(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    seen, kept = set(), []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            kept.append(s)
    return "\n".join(kept)


def main():
    parser = argparse.ArgumentParser(description="VLM Ablation Study")
    parser.add_argument("--model",   type=str, required=True, help="Model key, e.g. qwen3-4b")
    parser.add_argument("--setting", type=str, required=True, help="Ablation setting")
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = cfg["VLM_REGISTRY"]
    settings = cfg["ABLATION_SETTINGS"]
    if args.model not in registry:
        raise ValueError(f"--model must be one of {list(registry)}")
    if args.setting not in settings:
        raise ValueError(f"--setting must be one of {settings}")

    input_folder = cfg["ICT_OUTPUT_DIR"]
    global_csv   = cfg["GLOBAL_MSG_CSV"]
    output_dir   = cfg["ABLATION_OUTPUT_DIR"]
    cache_root   = cfg["ABLATION_CACHE_DIR"]
    image_size   = int(cfg.get("IMAGE_SIZE", 512))
    max_tokens   = int(cfg.get("MAX_NEW_TOKENS", 1024))

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.model}_{args.setting}.csv")
    if os.path.exists(output_file):
        print(f"[INFO] Output already exists, skipping: {output_file}")
        return

    global_df = pd.read_csv(global_csv)
    model, processor = load_vlm_model(args.model, registry[args.model])

    outputs, error_rows = [], []
    for fname in tqdm(sorted(os.listdir(input_folder))):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(input_folder, fname)
        try:
            (img_paths, captions, cap_texts, titles, _, _,
             sub_images_list, sub_types_list) = get_info(fpath)
        except Exception as e:
            error_rows.append({"input_json": fname, "img_index": -1,
                               "stage": "get_info", "error": str(e)})
            continue

        title = titles[0] if titles else ""
        row = global_df[global_df["literature_title"] == title]
        if row.empty:
            print(f"[WARN] No global_message for: {title!r}, skipping.")
            continue
        global_message = row["global_message"].values[0]

        one_outputs = []
        for idx in range(len(img_paths)):
            try:
                sub_types = sub_types_list[idx] or []
                n = len(sub_types)
                seg_desc = build_seg_description(sub_types, n)
                prompt = build_prompt(
                    args.setting, global_message,
                    captions[idx], cap_texts[idx], n, seg_desc,
                ).replace("\x00", "")

                all_imgs = [img_paths[idx]] + list(sub_images_list[idx] or [])
                img_array = [resize_image(p, image_size, cache_root) for p in all_imgs]

                actual_imgs = [] if args.setting == "txt_only" else img_array
                raw = run_vlm(args.model, model, processor, actual_imgs, prompt, max_tokens)
                one_outputs.append(deduplicate(raw))
            except Exception as e:
                err = str(e)
                print(f"[ERROR] {fname} idx={idx}: {err}")
                one_outputs.append(f"[ERROR] {err}")
                error_rows.append({"input_json": fname, "img_index": idx,
                                   "stage": "per_image", "error": err})

        outputs.append({"input_json": fname, "local_outputs": one_outputs})

    pd.DataFrame(outputs).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {output_file}")
    if error_rows:
        err_path = output_file.replace(".csv", "_errors.csv")
        pd.DataFrame(error_rows).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Errors saved: {err_path}")


if __name__ == "__main__":
    main()

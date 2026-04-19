"""
benchmark/cosine_eval.py
对 ablation CSV 的所有 model × setting 组合，用 SciBERT 计算 top-k 余弦相似度。

Usage:
    python -m benchmark.cosine_eval
"""

import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def make_embedder(model_path: str):
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path, use_safetensors=False).to(device)
    mdl.eval()

    def embed(text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(768)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = mdl(**inputs)
        pooled = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        vec = pooled.squeeze(0).cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-12)

    return embed


def build_reference_index(json_dir: Path) -> dict:
    index = {}
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(json_dir / fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            index[fname] = [item.get("img_cap_text", "") for item in data]
        except Exception as e:
            print(f"[WARN] failed to read {fname}: {e}")
    return index


def process_one_csv(csv_path: Path, model_name: str, setting: str,
                    ref_index: dict, embed, top_k: int = 5) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name}|{setting}"):
        input_json = row["input_json"]
        refs = ref_index.get(input_json, [])
        try:
            outputs = ast.literal_eval(row["local_outputs"])
        except Exception:
            outputs = [str(row["local_outputs"])]

        for fig_idx, model_output in enumerate(outputs):
            model_output = str(model_output)
            if model_output.startswith("[ERROR]"):
                rows.append({
                    "input_json": input_json, "fig_idx": fig_idx,
                    "model": model_name, "setting": setting,
                    "rank": -1, "cosine": np.nan,
                    "evidence_text": "", "model_output": model_output, "status": "error",
                })
                continue

            ref_text = refs[fig_idx] if fig_idx < len(refs) else ""
            q_vec = embed(model_output)

            sentences = [s.strip() for s in (ref_text or "").split("\n") if s.strip()] or [""]
            e_vecs = np.stack([embed(s) for s in sentences])
            sims = e_vecs @ q_vec
            topk = min(top_k, len(sentences))
            top_idx = sims.argsort()[::-1][:topk]

            for rank, i in enumerate(top_idx, start=1):
                rows.append({
                    "input_json": input_json, "fig_idx": fig_idx,
                    "model": model_name, "setting": setting,
                    "rank": rank, "cosine": float(sims[i]),
                    "evidence_text": sentences[i],
                    "model_output": model_output[:300],
                    "status": "ok",
                })
    return pd.DataFrame(rows)


def main():
    cfg = load_config("configs/config.yaml")
    scibert_path = cfg["SCIBERT_PATH"]
    ablation_dir = Path(cfg["ABLATION_OUTPUT_DIR"])
    json_dir     = Path(cfg["ICT_OUTPUT_DIR"])
    output_dir   = Path(cfg["COSINE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models   = list(cfg["VLM_REGISTRY"].keys())
    all_settings = cfg["ABLATION_SETTINGS"]

    embed = make_embedder(scibert_path)
    print("[INFO] Building reference index...")
    ref_index = build_reference_index(json_dir)
    print(f"[INFO] Loaded {len(ref_index)} files")

    all_results = []
    for model in all_models:
        for setting in all_settings:
            csv_path = ablation_dir / f"{model}_{setting}.csv"
            if not csv_path.exists():
                continue
            out_path = output_dir / f"{model}_{setting}_cosine.csv"
            if out_path.exists():
                all_results.append(pd.read_csv(out_path))
                continue
            print(f"\n[Processing] {model} | {setting}")
            df = process_one_csv(csv_path, model, setting, ref_index, embed)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            all_results.append(df)

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged_path = output_dir / "all_cosine_merged.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"[Saved merged] {merged_path}")

        summary = (merged[merged["rank"] == 1]
                   .groupby(["model", "setting"])["cosine"]
                   .mean().unstack("setting").round(4))
        print("\n===== Summary (Mean Cosine @ Rank 1) =====")
        print(summary.to_string())


if __name__ == "__main__":
    main()

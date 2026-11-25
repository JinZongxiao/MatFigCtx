import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def make_embedder(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path).to(device)
    mdl.eval()

    @torch.no_grad()
    def embed(text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(768)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = mdl(**inp)
        mask = inp["attention_mask"].unsqueeze(-1).float()
        vec = ((out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).squeeze(0).cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-12)

    return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scibert", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--sis-json", type=Path, required=True)
    ap.add_argument("--ict-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    embed = make_embedder(args.scibert)
    sis_map = json.loads(args.sis_json.read_text(encoding="utf-8"))

    rows = []
    for p in sorted(args.ict_dir.glob("*_image_captions_with_texts.json")):
        doc = p.name.replace("_image_captions_with_texts.json", "")
        recs = json.loads(p.read_text(encoding="utf-8"))
        for idx, rec in enumerate(recs):
            fid = f"{doc}::fig_{idx}"
            if fid not in sis_map:
                continue
            stmts = [x["statement"] if isinstance(x, dict) else x for x in sis_map[fid]]
            ref = f"{rec.get('img_caption', '')}\n{rec.get('img_cap_text', '')}".strip()
            if not stmts or not ref:
                continue
            sim = float(embed("\n".join(stmts)) @ embed(ref))
            rows.append({"id": fid, "doc": doc, "cosine": sim})

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Mean SciBERT cosine ({len(df)} figures): {df['cosine'].mean():.4f}")


if __name__ == "__main__":
    main()

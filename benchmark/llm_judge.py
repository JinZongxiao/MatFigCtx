"""
benchmark/llm_judge.py
Use an OpenAI-compatible LLM judge to score model outputs against evidence.

Dimensions:
- scientific_specificity
- visual_description_quality
- factual_plausibility
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config
from utils.llm_backends import make_openai_client


SYSTEM_PROMPT = """You are an expert evaluator of materials-science figure interpretations.
Score the candidate output against the provided evidence on three dimensions from 1 to 5:
- scientific_specificity
- visual_description_quality
- factual_plausibility

Return JSON only:
{
  "scientific_specificity": 1-5,
  "visual_description_quality": 1-5,
  "factual_plausibility": 1-5,
  "overall": 1-5,
  "rationale": "one short paragraph"
}
"""


def read_evidence(ict_path: Path) -> list[dict]:
    with ict_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge on model outputs")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input-csv", required=True, help="CSV with input_json and local_outputs columns")
    parser.add_argument("--ict-dir", help="Directory containing *_image_captions_with_texts.json files")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ict_dir = Path(args.ict_dir or cfg.get("ICT_OUTPUT_DIR", "./w_dataset_ict"))
    client, llm_cfg = make_openai_client(cfg, provider=args.provider, default="ernie")

    df = pd.read_csv(args.input_csv)
    rows = []
    for _, row in df.iterrows():
        input_json = row["input_json"]
        evidence_path = ict_dir / input_json
        if not evidence_path.exists():
            continue
        evidence_items = read_evidence(evidence_path)
        try:
            outputs = ast.literal_eval(row["local_outputs"])
        except Exception:
            outputs = [str(row["local_outputs"])]

        for fig_idx, output in enumerate(outputs):
            if fig_idx >= len(evidence_items):
                continue
            evidence = evidence_items[fig_idx]
            prompt = (
                f"Figure caption:\n{evidence.get('img_caption', '')}\n\n"
                f"Body text:\n{evidence.get('img_cap_text', '')}\n\n"
                f"Global context:\n{evidence.get('abstract', '')}\n{evidence.get('conclusion', '')}\n\n"
                f"Candidate output:\n{output}"
            )
            resp = client.chat.completions.create(
                model=llm_cfg["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {
                    "scientific_specificity": None,
                    "visual_description_quality": None,
                    "factual_plausibility": None,
                    "overall": None,
                    "rationale": raw,
                }
            result.update({
                "input_json": input_json,
                "fig_idx": fig_idx,
                "model_output": str(output),
                "judge_provider": llm_cfg["provider"],
                "judge_model": llm_cfg["model"],
            })
            rows.append(result)

    pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved judge results to {args.output_csv}")


if __name__ == "__main__":
    main()

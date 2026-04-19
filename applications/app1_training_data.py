"""
applications/app1_training_data.py
应用1：从材料论文PDF自动生成高质量SIS训练数据

Pipeline:
PDF → MinerU解析 → 证据链抽取 → VLM生成SIS → 训练数据JSON
"""

import json
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pdf_parser import parse_all_pdfs
from core.evidence_extractor import extract_all
from core.vlm_inference import load_vlm, generate_sis


def run(config_path: str = None, pdf_dir: str = None,
        output_dir: str = None, skip_parse: bool = False):
    """
    参数:
        config_path: YAML配置文件路径
        pdf_dir:     PDF输入目录（覆盖config）
        output_dir:  SIS输出目录（覆盖config）
        skip_parse:  跳过PDF解析（已有w_dataset）
    """
    # 加载配置
    cfg = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    pdf_dir        = pdf_dir or cfg.get("PDF_INPUT_DIR", "")
    paper_out_dir  = cfg.get("PAPER_OUTPUT_DIR", "./w_dataset")
    ict_dir        = cfg.get("ICT_OUTPUT_DIR",   "./w_dataset_ict")
    sis_dir        = output_dir or cfg.get("SIS_OUTPUT_DIR", "./sis_outputs")
    clf_path       = cfg.get("CLASSIFIER_PATH",  "")
    vlm_path       = cfg.get("VLM_MODEL_PATH",   "")

    os.makedirs(sis_dir, exist_ok=True)

    # Step 1: PDF解析
    if not skip_parse:
        print("\n[Step 1] Parsing PDFs with MinerU...")
        parse_all_pdfs(pdf_dir, paper_out_dir)
    else:
        print("[Step 1] Skipped PDF parsing")

    # Step 2: 证据链抽取
    print("\n[Step 2] Extracting evidence chains...")
    extract_all(paper_out_dir, ict_dir, clf_path)

    # Step 3: VLM生成SIS
    print("\n[Step 3] Generating SIS with VLM...")
    print(f"  Loading model: {vlm_path}")
    model, processor = load_vlm(vlm_path)

    ict_files = list(Path(ict_dir).glob("*_image_captions_with_texts.json"))
    print(f"  Found {len(ict_files)} papers to process")

    all_samples = []
    for ict_file in tqdm(ict_files, desc="Generating SIS"):
        out_file = Path(sis_dir) / ict_file.name.replace(
            "_image_captions_with_texts.json", "_sis.json"
        )
        if out_file.exists():
            continue

        with open(ict_file) as f:
            evidence_list = json.load(f)

        paper_samples = []
        for evidence in evidence_list:
            try:
                sis = generate_sis(evidence, model, processor)
                sample = {
                    "image":    evidence["img_path"],
                    "caption":  evidence["img_caption"],
                    "sis":      sis,
                    "source":   str(ict_file.name),
                }
                paper_samples.append(sample)
                all_samples.append(sample)
            except Exception as e:
                print(f"  [WARN] {ict_file.name}: {e}")

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(paper_samples, f, ensure_ascii=False, indent=2)

    # 合并输出
    merged_path = Path(sis_dir) / "all_training_data.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] {len(all_samples)} SIS samples saved to {merged_path}")
    return all_samples


if __name__ == "__main__":
    run(
        config_path="configs/config.yaml",
        skip_parse=True,   # 已有w_dataset时跳过解析
    )

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from core.evidence_extractor import extract_all
from core.pdf_parser import parse_all_pdfs
from core.visual_verifier import make_api_client, verify_visual_statements
from core.vlm_inference import extract_sis_three_stage, load_vlm
from utils.config import load_config


def main():
    ap = argparse.ArgumentParser(description="MatFigCtx three-stage SIS extraction pipeline")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--pdf-dir", default=None)
    ap.add_argument("--ict-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument("--skip-evidence", action="store_true")
    ap.add_argument("--verify", action="store_true", help="run cross-family visual verification filter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pdf_dir = args.pdf_dir or cfg["PDF_INPUT_DIR"]
    parsed_dir = cfg["PAPER_OUTPUT_DIR"]
    ict_dir = Path(args.ict_dir or cfg["ICT_OUTPUT_DIR"])
    out_dir = Path(args.out_dir or cfg["SIS_OUTPUT_DIR"])
    vlm_path = args.model or cfg["VLM_MODEL_PATH"]
    max_new_tokens = int(cfg.get("MAX_NEW_TOKENS", 1536))

    if not args.skip_parse:
        parse_all_pdfs(pdf_dir, parsed_dir)

    model, processor = load_vlm(vlm_path)

    if not args.skip_evidence:
        extract_all(parsed_dir, str(ict_dir), vlm=(model, processor))

    verifier = make_api_client(cfg.get("LLM_MODEL"), cfg.get("LLM_BASE_URL")) if args.verify else None

    out_dir.mkdir(parents=True, exist_ok=True)
    ict_files = sorted(ict_dir.glob("*_image_captions_with_texts.json"))
    all_sis = {}

    for ict_file in tqdm(ict_files, desc="Extracting SIS"):
        doc = ict_file.name.replace("_image_captions_with_texts.json", "")
        out_file = out_dir / f"{doc}_sis.json"
        if out_file.exists():
            continue

        evidence_list = json.loads(ict_file.read_text(encoding="utf-8"))
        doc_records = []
        for idx, ev in enumerate(evidence_list):
            fid = f"{doc}::fig_{idx}"
            stmts = extract_sis_three_stage(ev, model, processor, max_new_tokens=max_new_tokens, seed=args.seed)
            if verifier is not None:
                img_paths = [p for p in [ev.get("img_path", "")] + list(ev.get("sub_images") or []) if p and Path(p).exists()]
                stmts = verify_visual_statements(stmts, img_paths, ev.get("img_caption", ""), *verifier)
            doc_records.append({"id": fid, "image": ev["img_path"], "caption": ev["img_caption"], "statements": stmts})
            all_sis[fid] = stmts

        out_file.write_text(json.dumps(doc_records, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_sis:
        (out_dir / "all_sis.json").write_text(json.dumps(all_sis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()

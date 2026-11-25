import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from core.vlm_inference import _run_vlm, load_vlm
from core.visual_verifier import ask_json, make_api_client

CONDITIONS = ("zero_shot", "sis", "caption", "text", "text_len", "wrong_sis", "imgonly_sis")
SYSTEM_PROMPT = "You are a materials science expert. Answer the multiple-choice question with the option letter only."

LEAKAGE_PROMPT = """Below are a multiple-choice question about a figure in a materials-science paper, and a reference text.
You cannot see the figure. Answer ONLY from what the reference text says; do not use outside knowledge.
If the reference does not settle the answer, answer NONE.

[REFERENCE]
{reference}

[QUESTION]
{question}

[OPTIONS]
{options}

Return JSON only: {{"answer": "A|B|C|D|NONE", "quote": "supporting sentence or empty"}}"""


def parse_option_letter(value) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    m = (
        re.search(r"^\s*\(?([ABCD])\)?(?:[\.\):\s]|$)", text)
        or re.search(r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([ABCD])\)?", text)
        or re.search(r"\b([ABCD])\b", text)
    )
    return m.group(1) if m else text[:1]


def build_prompt(question: str, options, reference: str = None) -> str:
    opt_text = "\n".join(options) if isinstance(options, list) else str(options or "")
    parts = []
    if reference:
        parts.append(f"[Reference]\n{reference}")
    parts.append(f"[Question]\n{question}\n\n[Options]\n{opt_text}\n\nAnswer with the option letter only.")
    return "\n\n".join(parts)


def _gen_one(model, processor, img_path: Path, prompt: str, seed: int, max_new_tokens: int = 64) -> str:
    img = Image.open(img_path).convert("RGB")
    if processor is None:
        return _run_vlm(model, None, f"{SYSTEM_PROMPT}\n\n{prompt}", [img], max_new_tokens=max_new_tokens, seed=seed)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    kw = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if seed > 0:
        torch.manual_seed(seed)
        kw.update(do_sample=True, temperature=0.7, top_p=0.8)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def run_eval(args):
    refs = json.loads(args.refs_json.read_text(encoding="utf-8"))
    raw_items = [json.loads(l) for l in args.matcha_jsonl.open(encoding="utf-8")]
    items = []
    for idx, d in enumerate(raw_items):
        url = (d.get("article_info") or {}).get("article_url", "")
        art = url.split("/")[-1] if url else ""
        matched = next((k for k in refs if art and art in k), None)
        if not matched:
            continue
        vqa = (d.get("vqa") or [{}])[0]
        items.append({
            "idx": idx,
            "article": matched,
            "topic": vqa.get("topic", ""),
            "question": vqa.get("question", ""),
            "gt_answer": vqa.get("answer", ""),
            "options": vqa.get("options"),
            "image_path": args.img_root / (d.get("images") or [{}])[0].get("image_path", ""),
        })

    model, processor = load_vlm(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cond in args.conditions.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            out_csv = args.out_dir / f"{args.model_tag}_{cond}_seed{seed}.csv"
            if out_csv.exists():
                continue
            rows = []
            for it in tqdm(items, desc=f"{cond} s{seed}"):
                ref = None if cond == "zero_shot" else refs[it["article"]].get(cond)
                pred = _gen_one(model, processor, it["image_path"], build_prompt(it["question"], it["options"], ref), seed)
                rows.append({
                    "idx": it["idx"],
                    "article": it["article"],
                    "topic": it["topic"],
                    "gt": parse_option_letter(it["gt_answer"]),
                    "pred_raw": pred,
                    "pred": parse_option_letter(pred),
                    "correct": int(parse_option_letter(pred) == parse_option_letter(it["gt_answer"])),
                })
            pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
            acc = 100 * sum(r["correct"] for r in rows) / max(1, len(rows))
            print(f"{out_csv.name}: {acc:.2f}%")


def run_leakage_audit(args):
    refs = json.loads(args.refs_json.read_text(encoding="utf-8"))
    raw_items = [json.loads(l) for l in args.matcha_jsonl.open(encoding="utf-8")]
    done = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    client, model = make_api_client()

    todo = []
    for idx, d in enumerate(raw_items):
        if str(idx) in done:
            continue
        url = (d.get("article_info") or {}).get("article_url", "")
        art = url.split("/")[-1] if url else ""
        matched = next((k for k in refs if art and art in k), None)
        if not matched:
            continue
        vqa = (d.get("vqa") or [{}])[0]
        todo.append((idx, matched, vqa.get("question", ""), vqa.get("options"), parse_option_letter(vqa.get("answer", ""))))

    def work(job):
        idx, art, q, opts, gt = job
        try:
            ref = refs[art].get(args.ref_field, "")
            res = ask_json(client, model, LEAKAGE_PROMPT.format(reference=ref, question=q, options=opts), max_tokens=400)
            ans = str(res.get("answer", "NONE")).strip().upper()[:4]
            return str(idx), {"answer": ans, "gt": gt, "leaked": ans[:1] == gt, "quote": str(res.get("quote", ""))[:300]}
        except Exception as e:
            print(f"[ERR] {idx}: {e}", flush=True)
            return str(idx), None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for n, (k, res) in enumerate(ex.map(work, todo), 1):
            if res is not None:
                done[k] = res
            if n % 25 == 0 or n == len(todo):
                args.out.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    leaked = sum(v["leaked"] for v in done.values())
    print(f"Answer-in-reference: {leaked}/{len(done)} ({100 * leaked / max(1, len(done)):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval")
    ev.add_argument("--model", required=True)
    ev.add_argument("--model-tag", required=True)
    ev.add_argument("--matcha-jsonl", type=Path, required=True)
    ev.add_argument("--img-root", type=Path, required=True)
    ev.add_argument("--refs-json", type=Path, required=True)
    ev.add_argument("--out-dir", type=Path, required=True)
    ev.add_argument("--conditions", default=",".join(CONDITIONS))
    ev.add_argument("--seeds", default="0,1,2,3")

    lk = sub.add_parser("leakage")
    lk.add_argument("--matcha-jsonl", type=Path, required=True)
    lk.add_argument("--refs-json", type=Path, required=True)
    lk.add_argument("--ref-field", default="sis")
    lk.add_argument("--out", type=Path, required=True)
    lk.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "eval":
        run_eval(args)
    else:
        run_leakage_audit(args)


if __name__ == "__main__":
    main()

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from core.vlm_inference import _run_vlm, load_vlm, resize_keep_aspect

TOP_K = 8
WINDOW = 120
STRIDE = 100
MAX_IMAGES = 2
ARMS = ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5")

STOP = set(
    "a an the of in on for to and or with by at from as is are was were be been this that these those "
    "which what how does do did its it their there than into within between during after before under over".split()
)

INSTRUCT = (
    "Answer the question in one or two sentences. Use only the passages{img} given below. "
    "If they do not contain the answer, reply exactly: cannot answer."
)
CLOSED_PROMPT = (
    "Answer the question in one or two sentences from your own knowledge. "
    "If you do not know, reply exactly: cannot answer."
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]
        self.index = defaultdict(list)
        for i, d in enumerate(self.tf):
            for t in d:
                self.index[t].append(i)

    def top(self, query: list[str], k: int = TOP_K) -> list[int]:
        score = defaultdict(float)
        for t in set(query):
            for i in self.index.get(t, ()):
                f, n = self.tf[i][t], len(self.docs[i])
                score[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * n / self.avg))
        return sorted(score, key=lambda i: -score[i])[:k]


def sliding_windows(text: str) -> list[str]:
    w = (text or "").split()
    if not w:
        return []
    return [" ".join(w[i:i + WINDOW]) for i in range(0, max(1, len(w) - WINDOW + STRIDE), STRIDE)]


def build_text_chunks(ict_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    chunks, seen, fig_map = [], {}, {}
    for p in sorted(ict_dir.glob("*_image_captions_with_texts.json")):
        doc = p.name.replace("_image_captions_with_texts.json", "")
        recs = json.loads(p.read_text(encoding="utf-8"))
        if not recs:
            continue
        parts = [
            ("abstract", recs[0].get("abstract") or "", []),
            ("conclusion", recs[0].get("conclusion") or "", []),
        ]
        for idx, rec in enumerate(recs):
            fid = f"{doc}::fig_{idx}"
            fig_map[fid] = rec
            parts.append(("caption", rec.get("img_caption") or "", [fid]))
            parts.append(("paragraph", rec.get("img_cap_text") or "", [fid]))

        for kind, text, figs in parts:
            spans = [text.strip()] if kind == "caption" else sliding_windows(text)
            for w in spans:
                if not w:
                    continue
                key = (doc, w)
                if key in seen:
                    seen[key]["figures"] = sorted(set(seen[key]["figures"]) | set(figs))
                    continue
                seen[key] = {"cid": f"t{len(chunks)}", "doc": doc, "kind": kind, "figures": list(figs), "text": w}
                chunks.append(seen[key])
    return chunks, fig_map


def build_sis_chunks(sis_jsonl: Path, tag: str = "s") -> list[dict]:
    by_fig, caps = defaultdict(list), {}
    for line in sis_jsonl.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("reason"):
            continue
        caps[r["id"]] = r.get("caption") or ""
        by_fig[r["id"]].append(r["statement"])
    return [
        {"cid": f"{tag}{k}", "doc": fid.split("::fig_")[0], "kind": "sis", "figures": [fid], "text": caps[fid] + "\n" + "\n".join(s)}
        for k, (fid, s) in enumerate(by_fig.items())
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ict-dir", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--sis", type=Path, required=True)
    ap.add_argument("--sis-wrong", type=Path, default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    text_pool, fig_map = build_text_chunks(args.ict_dir)
    pools = {
        "Q1": text_pool,
        "Q2": text_pool + build_sis_chunks(args.sis, "s"),
        "Q4": text_pool,
    }
    if args.sis_wrong:
        pools["Q3"] = text_pool + build_sis_chunks(args.sis_wrong, "p")

    indices = {arm: BM25([tokenize(c["text"]) for c in pool]) for arm, pool in pools.items()}
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    model, processor = load_vlm(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms.split(","):
        if arm in ("Q1", "Q2", "Q3", "Q4") and arm not in pools:
            continue
        out_path = args.out_dir / f"{arm}.jsonl"
        done = {json.loads(l)["qid"] for l in out_path.open(encoding="utf-8")} if out_path.exists() else set()
        with out_path.open("a", encoding="utf-8") as f:
            for q in questions:
                if q["qid"] in done:
                    continue
                t0 = time.time()
                imgs, retrieved = [], []
                if arm == "Q0":
                    prompt = f"{CLOSED_PROMPT}\n\nQuestion: {q['question']}"
                elif arm == "Q5":
                    rec = fig_map[q["corpus_item"]]
                    if Path(rec["img_path"]).exists():
                        imgs = [resize_keep_aspect(rec["img_path"])]
                    ctx = f"{rec.get('img_caption', '')}\n{rec.get('img_cap_text', '')}".strip()
                    prompt = f"{INSTRUCT.format(img=' and figure')}\n\n{ctx}\n\nQuestion: {q['question']}"
                else:
                    hits = [pools[arm][i] for i in indices[arm].top(tokenize(q["question"]), TOP_K)]
                    retrieved = [c["cid"] for c in hits]
                    if arm == "Q4":
                        seen_f = []
                        for c in hits:
                            for fid in c["figures"]:
                                if fid not in seen_f and fid in fig_map and Path(fig_map[fid]["img_path"]).exists():
                                    seen_f.append(fid)
                                if len(seen_f) >= MAX_IMAGES:
                                    break
                            if len(seen_f) >= MAX_IMAGES:
                                break
                        imgs = [resize_keep_aspect(fig_map[fid]["img_path"]) for fid in seen_f]
                    ctx = "\n\n".join(f"[Passage {k + 1}]\n{c['text']}" for k, c in enumerate(hits))
                    prompt = f"{INSTRUCT.format(img=' and figures' if imgs else '')}\n\n{ctx}\n\nQuestion: {q['question']}"

                ans = _run_vlm(model, processor, prompt, imgs, max_new_tokens=256, seed=0)
                rec_out = {
                    "qid": q["qid"],
                    "arm": arm,
                    "answer": ans,
                    "retrieved": retrieved,
                    "images": len(imgs),
                    "latency_s": round(time.time() - t0, 3),
                }
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                f.flush()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

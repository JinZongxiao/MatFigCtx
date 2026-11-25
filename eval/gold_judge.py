import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from core.visual_verifier import ask_json, make_api_client

JUDGE_PROMPT = """You are evaluating candidate statements against human gold facts for a materials science figure.
You are not told which system produced the candidate statements.
Matching rules:
- Paraphrase is fine. Values read from a figure may differ by up to ~5% of the axis range or scale bar; reversed trends or wrong threshold sides do not match.
- "FULL_MATCH": accurately captures the material/sample, condition, and specific observation or value.
- "PARTIAL_MATCH": captures the main entity or trend direction, but omits a condition, panel contrast, or exact value.
- "MISS": no candidate statement covers the gold fact.

[GOLD FACTS]
{gold}

[CANDIDATE STATEMENTS]
{stmts}

Return JSON only:
{{"gold": [{{"id": "G1", "match": "FULL_MATCH|PARTIAL_MATCH|MISS", "statements": ["S1"]}}, ...]}}"""


def judge_figure(gold_facts: list[str], candidates: list[str], client, model: str) -> dict:
    g_block = "\n".join(f"G{k}. {g}" for k, g in enumerate(gold_facts, 1))
    s_block = "\n".join(f"S{k}. {s}" for k, s in enumerate(candidates, 1)) or "(none)"
    return ask_json(client, model, JUDGE_PROMPT.format(gold=g_block, stmts=s_block), max_tokens=8192)


def cluster_bootstrap_ci(doc_scores: dict[str, list[tuple[float, float]]], n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    docs = list(doc_scores)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(docs, size=len(docs), replace=True)
        num = sum(v * w for d in picked for v, w in doc_scores[d])
        den = sum(w for d in picked for _, w in doc_scores[d])
        vals.append(num / den if den else 0.0)
    pt = sum(v * w for d in docs for v, w in doc_scores[d]) / max(1e-12, sum(w for d in docs for _, w in doc_scores[d]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def exact_mcnemar(hits_a: list[bool], hits_b: list[bool]) -> tuple[int, int, float]:
    rescued = sum((not a) and b for a, b in zip(hits_a, hits_b))
    lost = sum(a and (not b) for a, b in zip(hits_a, hits_b))
    n = rescued + lost
    p = float(binomtest(min(rescued, lost), n, 0.5).pvalue) if n else 1.0
    return rescued, lost, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-json", type=Path, required=True)
    ap.add_argument("--pred-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    gold = json.loads(args.gold_json.read_text(encoding="utf-8"))
    preds = json.loads(args.pred_json.read_text(encoding="utf-8"))
    done = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    client, model = make_api_client()
    todo = [fid for fid in gold if fid not in done]

    def work(fid):
        try:
            g_list = [x["en"] if isinstance(x, dict) else x for x in gold[fid]]
            c_list = [x["statement"] if isinstance(x, dict) else x for x in preds.get(fid, [])]
            return fid, judge_figure(g_list, c_list, client, model)
        except Exception as e:
            print(f"[ERR] {fid}: {e}", flush=True)
            return fid, None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for n, (fid, res) in enumerate(ex.map(work, todo), 1):
            if res is not None:
                done[fid] = res
            if n % 10 == 0 or n == len(todo):
                args.out.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judged {len(done)}/{len(gold)} figures -> {args.out}")


if __name__ == "__main__":
    main()

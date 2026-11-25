import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

PROMPT_KEYS = {
    "Matrix Component": "Matrix Chemical Name",
    "Matrix Abbreviation": "Matrix Chemical Abbreviation",
    "Filler Abbreviation": "Filler Chemical Abbreviation",
    "Filler PST": "Filler Particle Surface Treatment Chemical Name",
    "Filler Mass": "Filler Composition Mass",
    "Filler Volume": "Filler Composition Volume",
}
MAX_PRED_POINTS = 2000


class _NumpyWith:
    def __init__(self, choice):
        self._choice = choice
        self.random = self

    def choice(self, *a, **k):
        return self._choice(*a, **k)

    def __getattr__(self, name):
        return getattr(np, name)


def as_prediction(true: dict) -> dict:
    pred = copy.deepcopy(true)
    for old, new in PROMPT_KEYS.items():
        pred[new] = pred.pop(old, None)
    return pred


def load_matvix_eval(eval_dir: Path, fix_match: bool, fix_order: bool):
    sys.path.insert(0, str(eval_dir.resolve()))
    os.chdir(eval_dir)  # MatViX standardize.py loads synonyms/ via relative path
    sys.setrecursionlimit(20000)
    import eval_composition as comp
    import eval_property as prop
    import matching_algorithm as match
    import standardize as std

    if fix_match:
        # MatViX negates similarity twice before Hungarian min-cost matching
        prop.match_samples = lambda scores, a, b: match.match_samples([[i, j, -s] for i, j, s in scores], a, b)
    if fix_order:
        real_choice = np.random.choice

        def ordered_choice(n, size=None, *a, **k):
            return np.sort(real_choice(n, size, replace=False)) if size is not None else real_choice(n, *a, **k)

        prop.np = _NumpyWith(choice=ordered_choice)
    return comp, prop, std


def clean_pred(pred: dict, fix_keys: bool) -> dict:
    pred = copy.deepcopy(pred)
    for key in ("Filler Composition Mass", "Filler Composition Volume"):
        pred.setdefault(key, None)
    if fix_keys:
        for true_key, prompt_key in PROMPT_KEYS.items():
            if prompt_key in pred:
                pred.setdefault(true_key, pred[prompt_key])
    props = pred.get("Properties")
    props = [props] if isinstance(props, dict) else (props if isinstance(props, list) else [])
    kept = []
    for p in props:
        if not isinstance(p, dict):
            continue
        data = p.get("data")
        if isinstance(data, list):
            try:
                data = [[float(x[0]), float(x[1])] for x in data if isinstance(x, (list, tuple)) and len(x) == 2]
            except (TypeError, ValueError):
                data = None
            if data and len(data) > MAX_PRED_POINTS:
                data = data[:: len(data) // MAX_PRED_POINTS + 1]
        else:
            data = None
        kept.append({**p, "data": data})
    pred["Properties"] = kept
    return pred


def score_article(true_dir: Path, preds: list, comp, prop, std, fix_keys: bool) -> dict:
    trues = [std.standardize_property(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(true_dir.glob("*.json"))]
    out = {
        "n_true": len(trues),
        "n_pred": len(preds),
        "samples": [],
        "properties": [],
        "n_true_properties": sum(len(t.get("Properties") or []) for t in trues),
    }
    preds = [clean_pred(p, fix_keys) for p in preds if isinstance(p, dict)]
    if not preds:
        return out
    matched = comp.eval_composition(preds, trues, "pnc")["matched_compositions"]
    for pred, true, (correct, fp, fn) in matched:
        p = correct / (correct + fp) if correct + fp else 0.0
        r = correct / (correct + fn) if correct + fn else 0.0
        out["samples"].append({"f1": 2 * p * r / (p + r) if p + r else 0.0})
        out["properties"].extend(prop.eval_properties(true, pred, "pnc"))
    return out


def summarize(by_article: dict) -> dict:
    samples = [s for a in by_article.values() for s in a["samples"]]
    props = [p for a in by_article.values() for p in a["properties"]]
    n_true = sum(a["n_true"] for a in by_article.values())
    n_props = sum(a["n_true_properties"] for a in by_article.values())
    return {
        "articles": len(by_article),
        "F1": round(100 * sum(s["f1"] for s in samples) / max(1, n_true), 2),
        "Headers": round(100 * sum(p["levenshtein headers"] for p in props) / max(1, n_props), 2),
        "Curves": round(100 * sum(p["frechet data"] for p in props) / max(1, n_props), 2),
        "CAS": round(100 * sum(p["average"] for p in props) / max(1, n_props), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--true-dir", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fix-keys", action="store_true")
    ap.add_argument("--fix-match", action="store_true")
    ap.add_argument("--fix-order", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    np.random.seed(0)
    ids = json.loads(args.ids.read_text(encoding="utf-8"))
    comp, prop, std = load_matvix_eval(args.eval_dir, args.fix_match, args.fix_order)

    by_article = {}
    for aid in ids:
        tdir = args.true_dir / aid
        if args.self_test:
            preds = [as_prediction(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(tdir.glob("*.json"))]
        else:
            pfile = args.pred_dir / f"{aid}.json"
            preds = json.loads(pfile.read_text(encoding="utf-8")) if pfile.exists() else []
        by_article[aid] = score_article(tdir, preds, comp, prop, std, args.fix_keys)

    summary = summarize(by_article)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "by_article": by_article}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

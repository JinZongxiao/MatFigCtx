import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.visual_verifier import ask_json, make_api_client

RELATIONS = (
    "has_composition", "processed_by", "has_microstructure", "has_phase", "has_property",
    "shows_trend", "compared_to", "causes", "located_at", "other",
)

EXTRACT_PROMPT = """Convert the numbered sentences about one figure of a materials-science paper into knowledge-graph triples.
Rules:
- subject and object name the specific material, sample, phase, feature or condition (resolve pronouns from context; never use a panel letter or figure number as an entity).
- relation is one of: {relations}.
- keep numbers and units in the object verbatim; one fact per triple; do not add anything not stated.
- cite the sentence number(s) each triple comes from.
Return JSON only: {{"triples": [["subject", "relation", "object", [sentence numbers]], ...]}}

{sentences}"""

COVER_PROMPT = """Below are gold FACTS about one figure of a materials-science paper, and automatically extracted TRIPLES.
For each fact decide whether the triples express it:
- "full": triples capture the entity, property, and specific value/trend (paraphrase fine; read values within ~5%).
- "partial": same entity and property, but a key condition/value/direction is missing or vague.
- "none": otherwise.
Return JSON only: {{"facts": [{{"id": "F1", "match": "full|partial|none"}}, ...]}}

FACTS:
{facts}

TRIPLES:
{triples}"""


def _clean_text(text: str) -> str:
    # PDF control chars occasionally break JSON payloads
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text or "")


def extract_figure_triples(statements: list[dict], client, model: str) -> list[dict]:
    numbered = "\n".join(f"{k}. {_clean_text(s['statement'])}" for k, s in enumerate(statements, 1))
    prompt = EXTRACT_PROMPT.format(relations=", ".join(RELATIONS), sentences=numbered)
    out = ask_json(client, model, prompt, max_tokens=6000)

    triples = []
    for t in out.get("triples", []):
        if isinstance(t, list) and len(t) == 4 and isinstance(t[3], list):
            ids = [int(x) for x in t[3] if str(x).isdigit() and 1 <= int(x) <= len(statements)]
            steps = sorted({statements[k - 1].get("step", "text") for k in ids})
            triples.append({"s": str(t[0]), "r": str(t[1]), "o": str(t[2]), "src": ids, "steps": steps})
    return triples


def score_coverage(facts: list[str], triples: list[dict], client, model: str) -> list[dict]:
    if not facts:
        return []
    f_block = "\n".join(f"F{k}. {f}" for k, f in enumerate(facts, 1))
    t_block = "\n".join(f"({t['s']}; {t['r']}; {t['o']})" for t in triples) or "(none)"
    out = ask_json(client, model, COVER_PROMPT.format(facts=f_block, triples=t_block), max_tokens=2000)
    return out.get("facts", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sis-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    data = json.loads(args.sis_json.read_text(encoding="utf-8"))
    done = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    client, model = make_api_client()
    todo = [k for k in data if k not in done]

    def work(fig_id):
        try:
            stmts = data[fig_id]
            return fig_id, {"statements": stmts, "triples": extract_figure_triples(stmts, client, model)}
        except Exception as e:
            print(f"[ERR] {fig_id}: {e}", flush=True)
            return fig_id, None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for n, (fid, res) in enumerate(ex.map(work, todo), 1):
            if res is not None:
                done[fid] = res
            if n % 10 == 0 or n == len(todo):
                args.out.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")

    n_triples = sum(len(v["triples"]) for v in done.values())
    print(f"Saved {n_triples} triples across {len(done)} figures to {args.out}")


if __name__ == "__main__":
    main()

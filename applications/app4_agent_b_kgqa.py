"""
applications/app4_agent_b_kgqa.py
应用4：基于三元组覆盖约束的Agent B问答前端

Pipeline:
Question -> KG retrieval -> coverage estimation -> grounded answer or abstention
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config
from utils.llm_backends import make_openai_client


SYSTEM_PROMPT = """You answer materials-science questions using only the supplied triples.
If the triples do not provide enough evidence, say that the knowledge graph coverage is insufficient.
When you answer, cite the supporting triples by their ids in square brackets such as [T3].
Do not use external knowledge.
"""


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_+\-\.]+", text.lower())


def load_triples(triples_path: str) -> list[dict]:
    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    if not isinstance(triples, list):
        raise ValueError("Triple file must contain a JSON array")
    return triples


def score_triple(question: str, triple: dict) -> int:
    q_counts = Counter(normalize_tokens(question))
    text = " ".join(str(triple.get(k, "")) for k in ("subject", "predicate", "object", "caption", "source"))
    t_counts = Counter(normalize_tokens(text))
    return sum(min(q_counts[token], t_counts[token]) for token in q_counts)


def retrieve_triples(question: str, triples: list[dict], top_k: int = 5) -> list[dict]:
    scored = []
    for idx, triple in enumerate(triples):
        score = score_triple(question, triple)
        if score <= 0:
            continue
        enriched = dict(triple)
        enriched["_score"] = score
        enriched["_id"] = f"T{idx + 1}"
        scored.append(enriched)
    scored.sort(key=lambda x: (-x["_score"], x["_id"]))
    return scored[:top_k]


def estimate_coverage(question: str, retrieved: list[dict]) -> dict:
    q_tokens = set(normalize_tokens(question))
    if not q_tokens:
        return {"coverage": 0.0, "status": "insufficient"}
    covered = set()
    for triple in retrieved:
        covered.update(normalize_tokens(" ".join(str(triple.get(k, "")) for k in ("subject", "predicate", "object"))))
    overlap = len(q_tokens & covered)
    coverage = overlap / max(len(q_tokens), 1)
    status = "covered" if coverage >= 0.2 and retrieved else "insufficient"
    return {"coverage": round(coverage, 4), "status": status}


def answer_question(question: str, retrieved: list[dict], coverage: dict, client=None, model: str = "") -> dict:
    if coverage["status"] != "covered":
        return {
            "status": "insufficient_coverage",
            "answer": "Insufficient knowledge-graph coverage for a grounded answer.",
            "coverage": coverage["coverage"],
            "supporting_triples": retrieved,
        }

    if client is None or not model:
        return {
            "status": "covered_no_llm",
            "answer": "Coverage is sufficient, but no LLM client was provided. Use the supporting triples directly.",
            "coverage": coverage["coverage"],
            "supporting_triples": retrieved,
        }

    evidence_lines = []
    for triple in retrieved:
        evidence_lines.append(
            f"{triple['_id']}: ({triple.get('subject', '')}, {triple.get('predicate', '')}, {triple.get('object', '')})"
        )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Coverage estimate: {coverage['coverage']}\n\n"
        "Supporting triples:\n" + "\n".join(evidence_lines)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    answer = resp.choices[0].message.content.strip()
    return {
        "status": "success",
        "answer": answer,
        "coverage": coverage["coverage"],
        "supporting_triples": retrieved,
    }


def run_interactive(config_path: str = None, triples_path: str = None, provider: str = None):
    cfg = load_config(config_path or "configs/config.yaml")
    triples_path = triples_path or cfg.get("KG_OUTPUT_JSON", "./kg_triples.json")
    triples = load_triples(triples_path)
    client, llm_cfg = make_openai_client(cfg, provider=provider, default="ernie")

    print("\n[MatFigCtx Agent B] Ready. Type 'quit' to exit.\n")
    print(f"[KG] Loaded {len(triples)} triples from {triples_path}")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        retrieved = retrieve_triples(user_input, triples, top_k=int(cfg.get("AGENT_B_TOP_K", 5)))
        coverage = estimate_coverage(user_input, retrieved)
        result = answer_question(user_input, retrieved, coverage, client=client, model=llm_cfg["model"])
        print("\nAgent B:")
        print(result["answer"])
        print(f"\nCoverage: {result['coverage']}")
        print(f"Triples used: {len(result['supporting_triples'])}\n")


if __name__ == "__main__":
    run_interactive(config_path="configs/config.yaml")

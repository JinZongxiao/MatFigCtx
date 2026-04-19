"""
applications/app2_knowledge_graph.py
应用2：面向材料知识图谱的自动三元组抽取与Neo4j入库

Pipeline:
SIS → LLM三元组抽取 → Neo4j知识图谱
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config
from utils.llm_backends import make_openai_client

SYSTEM_PROMPT = """You are a materials science knowledge graph expert.
Extract structured triples from the given scientific statement about materials.

Output ONLY a valid JSON array of triples, no other text:
[
  {"subject": "...", "predicate": "...", "object": "..."},
  ...
]

Guidelines:
- subject: material, phase, structure, or process
- predicate: relationship verb (e.g., "has_grain_size", "improves", "contains", "forms_at")
- object: property value, material, phase, or condition
- Extract ALL meaningful triples from the statement
- Use consistent snake_case for predicates
- Include units in object when present (e.g., "580 nm")
"""


def extract_triples(sis_text: str, client, model: str) -> list:
    """从单条SIS抽取知识图谱三元组"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Scientific statement:\n{sis_text}"},
            ],
            stream=False,
        )
        raw = resp.choices[0].message.content.strip()
        # 清理markdown代码块
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [WARN] Triple extraction failed: {e}")
        return []


def write_to_neo4j(triples: list, metadata: dict, driver):
    """将三元组写入Neo4j"""
    from neo4j import GraphDatabase
    with driver.session() as session:
        for triple in triples:
            subj = triple.get("subject", "").strip()
            pred = triple.get("predicate", "").strip().replace(" ", "_")
            obj  = triple.get("object",  "").strip()
            if not subj or not pred or not obj:
                continue
            query = (
                f"MERGE (s:Entity {{name: $subj}}) "
                f"MERGE (o:Entity {{name: $obj}}) "
                f"MERGE (s)-[r:{pred} {{source: $source, caption: $caption}}]->(o)"
            )
            try:
                session.run(query,
                           subj=subj, obj=obj,
                           source=metadata.get("source", ""),
                           caption=metadata.get("caption", ""))
            except Exception as e:
                print(f"  [NEO4J WARN] {e}")


def run(config_path: str = None, sis_dir: str = None,
        output_json: str = None, provider: str = None):
    """
    参数:
        config_path: YAML配置文件路径
        sis_dir:     SIS JSON目录
        output_json: 三元组输出JSON路径（同时写入Neo4j）
    """
    cfg = load_config(config_path or "configs/config.yaml")
    sis_dir     = sis_dir or cfg.get("SIS_OUTPUT_DIR", "./sis_outputs")
    output_json = output_json or "./kg_triples.json"
    client, llm_cfg = make_openai_client(cfg, provider=provider, default="ernie")
    model_name  = llm_cfg["model"]
    neo4j_uri   = cfg.get("NEO4J_URI",        "bolt://localhost:7687")
    neo4j_user  = cfg.get("NEO4J_USER",       "neo4j")
    from utils.config import get_secret
    neo4j_pass  = get_secret(cfg, "NEO4J_PASSWORD_ENV", "password")

    # 连接Neo4j
    driver = None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri,
                                      auth=(neo4j_user, neo4j_pass))
        driver.verify_connectivity()
        print(f"[Neo4j] Connected: {neo4j_uri}")
    except Exception as e:
        print(f"[Neo4j] Connection failed: {e}")
        print("  → Will save triples to JSON only")

    # 处理SIS数据
    sis_files = list(Path(sis_dir).glob("*_sis.json"))
    print(f"[INFO] Found {len(sis_files)} SIS files")

    all_triples = []
    for sis_file in sis_files:
        with open(sis_file) as f:
            samples = json.load(f)

        for sample in samples:
            sis_text = sample.get("sis", "")
            if not sis_text:
                continue

            triples = extract_triples(sis_text, client, model_name)
            metadata = {
                "source":  sample.get("source", str(sis_file.name)),
                "caption": sample.get("caption", ""),
                "image":   sample.get("image",   ""),
            }

            for t in triples:
                t.update(metadata)
            all_triples.extend(triples)

            if driver:
                write_to_neo4j(triples, metadata, driver)

            time.sleep(0.2)

    # 保存JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)

    if driver:
        driver.close()

    print(f"\n[Done] {len(all_triples)} triples extracted")
    print(f"  → JSON: {output_json}")
    print(f"  → LLM backend: {llm_cfg['provider']} / {model_name}")
    return all_triples


if __name__ == "__main__":
    run(config_path="configs/config.yaml")

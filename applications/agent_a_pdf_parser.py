"""
Agent A — PDF Parser接口型（真正接入VLM）
Admin节点运行，自动提交GPU作业
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

MATFIGCTX_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, MATFIGCTX_ROOT)

from utils.config import load_config, get_secret

_CFG = load_config("configs/config.yaml")

BASE_URL       = _CFG.get("ERNIE_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
MODEL          = _CFG.get("ERNIE_MODEL",    "ernie-4.5-turbo-128k-preview")
VLM_MODEL_PATH = _CFG.get("VLM_MODEL_PATH", "")
PAPER_OUT_DIR  = _CFG.get("PAPER_OUTPUT_DIR", "")
ICT_DIR        = _CFG.get("ICT_OUTPUT_DIR",   "")
SIS_OUTPUT_DIR = _CFG.get("SIS_OUTPUT_DIR",   "")
CLF_PATH       = _CFG.get("CLASSIFIER_PATH",  "")
_client: Any = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = get_secret(_CFG, "ERNIE_API_KEY_ENV")
        if not api_key:
            raise RuntimeError(
                "ERNIE_API_KEY env var not set. "
                "Copy .env.example -> .env and fill in ERNIE_API_KEY."
            )
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client

# =========================
# VLM推理（仅在GPU节点调用）
# =========================
_vlm_model     = None
_vlm_processor = None

def get_vlm():
    global _vlm_model, _vlm_processor
    if _vlm_model is None:
        print(f"[VLM] Loading {VLM_MODEL_PATH}...")
        from core.vlm_inference import load_vlm
        _vlm_model, _vlm_processor = load_vlm(VLM_MODEL_PATH)
        print("[VLM] Model loaded")
    return _vlm_model, _vlm_processor


# =========================
# 工具实现
# =========================
def tool_extract_evidence_chain(paper_name: str) -> dict:
    ict_path = os.path.join(ICT_DIR, f"{paper_name}_image_captions_with_texts.json")
    if not os.path.exists(ict_path):
        cl_path = os.path.join(PAPER_OUT_DIR, paper_name, "markdowns", f"{paper_name}_content_list.json")
        md_path = os.path.join(PAPER_OUT_DIR, paper_name, "markdowns", f"{paper_name}.md")
        if not os.path.exists(cl_path):
            return {"status": "error", "error": f"Paper not found: {paper_name}."}
        from core.evidence_extractor import extract_evidence_chain
        evidence = extract_evidence_chain(cl_path, md_path, CLF_PATH)
        with open(ict_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, ensure_ascii=False, indent=2)
    with open(ict_path, encoding="utf-8") as f:
        data = json.load(f)
    summary = []
    for i, item in enumerate(data):
        types = [s.get("predicted_type", s.get("type", "unknown"))
                 for s in item.get("sub_image_type", [])]
        summary.append({
            "fig_index":   i,
            "caption":     item.get("img_caption", "")[:100],
            "n_subfigs":   len(item.get("sub_images", [])),
            "subfig_types": types,
        })
    return {
        "status":     "success",
        "paper_name": paper_name,
        "title":      data[0].get("literature_title", "") if data else "",
        "n_figures":  len(data),
        "figures":    summary,
    }


def tool_analyze_figure(paper_name: str, fig_index: int = 0) -> dict:
    """先查缓存，缓存没有则返回needs_gpu让Agent提交作业"""
    sis_path = os.path.join(SIS_OUTPUT_DIR,
                            f"{paper_name}_image_captions_with_texts_sis.json")
    if os.path.exists(sis_path):
        with open(sis_path, encoding="utf-8") as f:
            cached = json.load(f)
        if fig_index < len(cached) and cached[fig_index].get("sis"):
            return {
                "status":  "success",
                "sis":     cached[fig_index]["sis"],
                "caption": cached[fig_index].get("caption", ""),
                "source":  "cache",
            }
    # 没有缓存，需要GPU
    return {
        "status":     "needs_gpu",
        "message":    "SIS not cached. Please call submit_gpu_job to run VLM inference on GPU node.",
        "paper_name": paper_name,
        "fig_index":  fig_index,
    }


def tool_analyze_all_figures(paper_name: str) -> dict:
    """先查缓存，缓存没有则返回needs_gpu"""
    sis_path = os.path.join(SIS_OUTPUT_DIR,
                            f"{paper_name}_image_captions_with_texts_sis.json")
    if os.path.exists(sis_path):
        with open(sis_path, encoding="utf-8") as f:
            cached = json.load(f)
        results = [{"fig_index": i, "caption": d.get("caption", "")[:80], "sis": d.get("sis", "")}
                   for i, d in enumerate(cached) if d.get("sis")]
        if results:
            return {"status": "success", "n_figures": len(results), "results": results, "source": "cache"}
    return {
        "status":     "needs_gpu",
        "message":    "SIS not cached. Please call submit_gpu_job with fig_index=-1 to analyze all figures.",
        "paper_name": paper_name,
    }


def tool_submit_gpu_job(paper_name: str, fig_index: int = -1) -> dict:
    """生成sbatch脚本并提交到GPU节点"""
    log_dir = os.path.join(MATFIGCTX_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    job_name    = f"vlm_{paper_name[-12:].replace('-','')}"
    log_path    = os.path.join(log_dir, f"{job_name}.log")
    script_path = os.path.join(log_dir, f"{job_name}.sh")
    sis_out     = os.path.join(SIS_OUTPUT_DIR,
                               f"{paper_name}_image_captions_with_texts_sis.json")

    if fig_index == -1:
        py_task = f"""
from core.vlm_inference import generate_sis, load_vlm
import json, os

ict_path = "{os.path.join(ICT_DIR, paper_name + '_image_captions_with_texts.json')}"
with open(ict_path, encoding='utf-8') as f:
    evidence_list = json.load(f)

model, processor = load_vlm("{VLM_MODEL_PATH}")
results = []
for i, evidence in enumerate(evidence_list):
    print(f"  [VLM] Fig {{i+1}}/{{len(evidence_list)}}...")
    try:
        sis = generate_sis(evidence, model, processor)
        results.append({{"fig_index": i, "caption": evidence.get("img_caption","")[:80], "sis": sis}})
    except Exception as e:
        results.append({{"fig_index": i, "error": str(e)}})

with open("{sis_out}", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"[Done] {{len(results)}} figures saved to {sis_out}")
"""
    else:
        py_task = f"""
from core.vlm_inference import generate_sis, load_vlm
import json, os

ict_path = "{os.path.join(ICT_DIR, paper_name + '_image_captions_with_texts.json')}"
with open(ict_path, encoding='utf-8') as f:
    evidence_list = json.load(f)

evidence = evidence_list[{fig_index}]
model, processor = load_vlm("{VLM_MODEL_PATH}")
sis = generate_sis(evidence, model, processor)

sis_path = "{sis_out}"
cached = []
if os.path.exists(sis_path):
    with open(sis_path, encoding='utf-8') as f:
        cached = json.load(f)
while len(cached) <= {fig_index}:
    cached.append({{}})
cached[{fig_index}] = {{"fig_index": {fig_index}, "caption": evidence.get("img_caption",""), "sis": sis}}
with open(sis_path, "w", encoding="utf-8") as f:
    json.dump(cached, f, ensure_ascii=False, indent=2)
print(f"[Done] Fig {fig_index} saved.")
print("SIS:", sis[:300])
"""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --partition=GPU_Node
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

echo "[START] $(date)"
cd {MATFIGCTX_ROOT}
/public/home/normal_bgd/anaconda3/envs/qwen_env/bin/python << 'PYEOF'
import sys
sys.path.insert(0, "{MATFIGCTX_ROOT}")
{py_task}
PYEOF
echo "[DONE] $(date)"
"""

    with open(script_path, "w") as f:
        f.write(script)

    try:
        r      = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        job_id = r.stdout.strip().split()[-1] if r.returncode == 0 else None
        return {
            "status":    "submitted" if r.returncode == 0 else "error",
            "job_id":    job_id,
            "log_path":  log_path,
            "message":   r.stdout.strip() or r.stderr.strip(),
            "check_cmd": f"squeue | grep {job_id}",
            "log_cmd":   f"tail -f {log_path}",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def tool_read_job_result(paper_name: str, job_id: str = None, fig_index: int = -1) -> dict:
    """检查job状态并读取SIS结果"""
    job_status = "unknown"
    if job_id:
        r = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        job_status = "running" if job_id in r.stdout else "done"

    sis_path = os.path.join(SIS_OUTPUT_DIR,
                            f"{paper_name}_image_captions_with_texts_sis.json")
    if not os.path.exists(sis_path):
        return {
            "status":     "pending",
            "job_status": job_status,
            "message":    "SIS file not ready yet. Job may still be running.",
        }

    with open(sis_path, encoding="utf-8") as f:
        data = json.load(f)

    if fig_index == -1:
        results = [{"fig_index": i, "caption": d.get("caption", "")[:80], "sis": d.get("sis", "")}
                   for i, d in enumerate(data) if d.get("sis")]
    else:
        results = ([{"fig_index": fig_index,
                     "sis":       data[fig_index].get("sis", ""),
                     "caption":   data[fig_index].get("caption", "")}]
                   if fig_index < len(data) else [])

    return {
        "status":     "success",
        "job_status": job_status,
        "n_figures":  len(results),
        "results":    results,
    }


# =========================
# 工具注册（必须在__main__之前）
# =========================
TOOLS = [
    {"type": "function", "function": {
        "name": "extract_evidence_chain",
        "description": "Extract figure-text evidence chains from a parsed paper. Returns captions, body text, abstract, subfigure types.",
        "parameters": {"type": "object", "properties": {
            "paper_name": {"type": "string", "description": "Paper folder name"}
        }, "required": ["paper_name"]}
    }},
    {"type": "function", "function": {
        "name": "analyze_figure",
        "description": "Check SIS cache for a specific figure. Returns cached SIS if available, otherwise returns needs_gpu status.",
        "parameters": {"type": "object", "properties": {
            "paper_name": {"type": "string"},
            "fig_index":  {"type": "integer", "default": 0}
        }, "required": ["paper_name"]}
    }},
    {"type": "function", "function": {
        "name": "analyze_all_figures",
        "description": "Check SIS cache for all figures. Returns cached SIS if available, otherwise returns needs_gpu status.",
        "parameters": {"type": "object", "properties": {
            "paper_name": {"type": "string"}
        }, "required": ["paper_name"]}
    }},
    {"type": "function", "function": {
        "name": "submit_gpu_job",
        "description": "Submit a SLURM GPU job to run VLM inference. Use when analyze_figure/analyze_all_figures returns needs_gpu. fig_index=-1 means all figures.",
        "parameters": {"type": "object", "properties": {
            "paper_name": {"type": "string"},
            "fig_index":  {"type": "integer", "default": -1, "description": "-1 = all figures"}
        }, "required": ["paper_name"]}
    }},
    {"type": "function", "function": {
        "name": "read_job_result",
        "description": "Check job status and read SIS results after GPU job completes.",
        "parameters": {"type": "object", "properties": {
            "paper_name": {"type": "string"},
            "job_id":     {"type": "string"},
            "fig_index":  {"type": "integer", "default": -1}
        }, "required": ["paper_name"]}
    }},
]

TOOL_MAP = {
    "extract_evidence_chain": tool_extract_evidence_chain,
    "analyze_figure":         tool_analyze_figure,
    "analyze_all_figures":    tool_analyze_all_figures,
    "submit_gpu_job":         tool_submit_gpu_job,
    "read_job_result":        tool_read_job_result,
}


# =========================
# Agent主循环
# =========================
def run_agent_a(user_query: str, verbose: bool = True) -> str:
    client = get_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You are MatFigCtx Agent A, running on the ADMIN NODE (no GPU). "
                "You analyze materials science papers using a context-grounded VLM pipeline. "
                "Workflow: "
                "1) extract_evidence_chain — get paper structure. "
                "2) analyze_figure or analyze_all_figures — check SIS cache. "
                "3) If result is needs_gpu — call submit_gpu_job to submit SLURM job. "
                "4) call read_job_result to check if job is done and get SIS. "
                "5) If job still running, report job_id and log_cmd to user. "
                "NEVER try to load or run VLM yourself. Always use submit_gpu_job for VLM inference."
            )
        },
        {"role": "user", "content": user_query}
    ]

    if verbose:
        print(f"\n{'='*60}\n[User] {user_query}")

    for _ in range(10):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            if verbose:
                print(f"\n[Agent A]\n{msg.content}")
            return msg.content

        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            if verbose:
                print(f"\n[Tool] {fn_name}({json.dumps(fn_args)[:80]})")
            fn     = TOOL_MAP.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}
            if verbose:
                rs = json.dumps(result, ensure_ascii=False)
                print(f"  → {rs[:400]}{'...' if len(rs) > 400 else ''}")
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result, ensure_ascii=False),
            })

    return "Max rounds reached."


# =========================
# Demo
# =========================
if __name__ == "__main__":
    print("MatFigCtx Agent A — Interactive Mode")
    print("Type 'quit' to exit\n")
    while True:
        query = input("[You] ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if query:
            run_agent_a(query)

#    DEMO_PAPER = "1-s2.0-S0263436821002857-main"

#    run_agent_a(
#        f"Extract the evidence chain for paper {DEMO_PAPER} "
#        f"and tell me how many figures it has and what types they are."
#    )
#    run_agent_a(
#        f"Analyze the first figure (fig_index=0) of paper {DEMO_PAPER}. "
#        f"Submit a GPU job if needed."
#    )
#    run_agent_a(
#        f"Analyze ALL figures in paper {DEMO_PAPER}. "
#        f"Submit a GPU job if needed and tell me how to check progress."
#    )

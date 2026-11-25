import base64
import json
import os
import re
import time

from openai import OpenAI

VERIFY_PROMPT = """You are auditing scientific statements extracted from a materials characterization figure.
Image 1 is the full figure; any further images are its subfigures in order.
Paper caption is given for context only; judge each statement against what the figure actually shows.

Caption: {caption}

For each statement return:
- "SUPPORTED": directly visible and consistent with the figure (read values within ~5% of axis range, trend direction right).
- "CONTRADICTED": visually wrong value, trend, phase, or morphology.
- "CANNOT_TELL": requires non-visual context or detail is not readable.

[STATEMENTS]
{stmts}

Return JSON only: {{"statements": [{{"id": "S1", "verdict": "SUPPORTED|CONTRADICTED|CANNOT_TELL"}}, ...]}}"""


def make_api_client(model: str = None, base_url: str = None, api_key: str = None) -> tuple[OpenAI, str]:
    key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set LLM_API_KEY (or OPENAI_API_KEY / DEEPSEEK_API_KEY) in .env")
    url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    mdl = model or os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    return OpenAI(api_key=key, base_url=url, timeout=300.0), mdl


def ask_json(client: OpenAI, model: str, prompt: str, images: list[str] = (), max_tokens: int = 8192, tries: int = 4) -> dict:
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
        for p in images
    ] + [{"type": "text", "text": prompt}]

    extra = {}
    base_str = str(getattr(client, "base_url", "")).lower()
    if "deepseek" in model.lower() or "deepseek" in base_str:
        # deepseek-flash enables thinking by default, which eats into max_tokens
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    last_err = None
    for attempt in range(tries):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content if images else prompt}],
                **extra,
            )
            # LaTeX backslashes in statements (e.g. \alpha, \mu) break strict json.loads
            raw = re.sub(r'(\\\\)|\\(?!["/bfnrtu])', lambda m: m.group(1) or r"\\", r.choices[0].message.content)
            out = json.loads(raw)
            out["_served_model"] = r.model
            return out
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"API call failed after {tries} tries: {last_err}")


def verify_visual_statements(
    records: list[dict],
    image_paths: list[str],
    caption: str,
    client: OpenAI,
    model: str,
) -> list[dict]:
    visual = [(i, r) for i, r in enumerate(records) if r.get("step") in ("figure", "complement")]
    if not visual or not image_paths:
        return records

    stmt_block = "\n".join(f"S{k}. {r['statement']}" for k, (_, r) in enumerate(visual, 1))
    prompt = VERIFY_PROMPT.format(caption=caption, stmts=stmt_block)
    res = ask_json(client, model, prompt, images=image_paths)
    verdicts = {s["id"]: str(s.get("verdict", "CANNOT_TELL")).upper() for s in res.get("statements", [])}

    drop_idx = set()
    for k, (idx, r) in enumerate(visual, 1):
        v = verdicts.get(f"S{k}", "CANNOT_TELL")
        r["visual_verdict"] = v
        if v == "CONTRADICTED":
            drop_idx.add(idx)

    return [r for i, r in enumerate(records) if i not in drop_idx]

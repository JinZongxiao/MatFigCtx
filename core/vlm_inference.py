import base64
import io
import os
import re
from pathlib import Path

import torch
from PIL import Image

MAX_IMAGE_SIDE = 1024  # keep aspect ratio; 512x512 padding blurred axis ticks
MAX_LOCAL_CHARS = 4000
TAG_RE = re.compile(r"^\s*\[(FIG\+TXT|FIG|TXT)\]\s*(\((?:subfigure|sub-figure|panel)\s*[^)]*\))?\s*", re.I)

PROMPT_STAGE1_TEXT = (
    "You are an expert assistant in materials science literature understanding.\n"
    "Based on the following evidence, generate concise scientific statements describing the figure content.\n\n"
    "[Figure Caption]: {caption}\n"
    "[Body Text] (paragraphs directly referencing this figure): {body_text}\n"
    "(No image is provided. Generate statements based on text evidence only.)\n\n"
    "Output requirements:\n"
    "- Report every distinct fact the text states about this figure: materials and compositions, processing and "
    "test conditions, values, trends, comparisons and conclusions.\n"
    "- Write one fact per statement. Do not merge several facts into one sentence.\n"
    "- Each statement must be an independent scientific fact sentence (subject + verb + object).\n"
    "- Every statement must be supported by the evidence text above. Leave out facts about other figures.\n"
    "- State the fact itself; do not refer to the text, caption or figure numbers inside the statement.\n"
    "- Do NOT add numbering, explanations, or any other text.\n"
    "- Output scientific statements ONLY, one per line."
)

PROMPT_STAGE2_VISUAL = (
    "You are an expert assistant in materials science literature understanding.\n"
    "Based on the figure and the following evidence, generate concise scientific statements describing the figure content.\n\n"
    "[Images]: {image_labels}\n"
    "[Figure Caption]: {caption}\n"
    "[Body Text] (paragraphs directly referencing this figure): {body_text}\n\n"
    "Output requirements:\n"
    "- Write ONLY statements read from the figure itself. Use the caption and text only to identify which sample, "
    "phase or condition each subfigure shows. Do not repeat facts the text already states.\n"
    "- Work through the subfigures one at a time, according to their type:\n"
    "  * Plots and bar charts: find the axes, units and legend first. For each named series give its values at "
    "labelled points, its maximum or minimum, and its trend. Keep each series with its own axis.\n"
    "  * XRD patterns: give the labelled phases, which phases appear or vanish between samples, and peak shifts or broadening.\n"
    "  * Micrographs and maps: describe what is visible: grain shape (equiaxed, elongated), particle shape and location "
    "(inside grains, at grain boundaries), distribution (uniform, agglomerated), pores, cracks, fracture mode, element "
    "enrichment. Give sizes only when a scale bar allows it.\n"
    "  * Schematics: give the labelled parts and what they indicate.\n"
    "- Compare samples and panels: state which is larger, finer, denser, stronger, and how a quantity changes with "
    "composition, temperature or processing.\n"
    "- Name the sample, composition or condition in every statement (for example 'The W-1 wt% Y2O3 sample sintered "
    "at 1600 °C ...'), never only a panel letter.\n"
    "- Never write a statement that only names the image type, magnification or scale bar.\n"
    "- Give a number only if you can read it from a label or axis ticks. Never reuse the same number for different "
    "samples unless the figure shows it.\n"
    "- Start each statement with [FIG] and the subfigure in parentheses, e.g. [FIG] (subfigure 2) ...\n"
    "- Output scientific statements ONLY, one per line."
)

PROMPT_STAGE3_COMPLEMENT = (
    "You are an expert assistant in materials science literature understanding.\n"
    "Based on the figure and the following evidence, generate concise scientific statements describing the figure content.\n\n"
    "[Images]: {image_labels}\n"
    "[Figure Caption]: {caption}\n"
    "[Body Text] (paragraphs directly referencing this figure): {body_text}\n"
    "[Facts already extracted from the text]:\n{text_facts}\n\n"
    "Output requirements:\n"
    "- The list [Facts already extracted from the text] above was extracted from the paper. Write ONLY what the figure "
    "shows and that list does not contain: values read from axes, trends, comparisons between samples or panels, and "
    "visible features. If the figure adds a number or a detail to a listed fact, write the new part. Never restate a listed fact.\n"
    "- Work through the subfigures one at a time (plots/bar charts, XRD patterns, micrographs/maps, schematics), naming "
    "the sample, composition or condition in every statement rather than only a panel letter.\n"
    "- Never write a statement that only names the image type, magnification or scale bar.\n"
    "- Give a number only if you can read it from a label or axis ticks.\n"
    "- Start each statement with [FIG] and the subfigure in parentheses, e.g. [FIG] (subfigure 2) ...\n"
    "- Output scientific statements ONLY, one per line."
)


class OpenAIVLMBackend:
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        from openai import OpenAI
        key = api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        url = base_url or os.getenv("VLM_API_BASE_URL")
        self.client = OpenAI(api_key=key, base_url=url, timeout=300.0)
        self.model_name = model_name


def load_vlm(model_spec: str):
    if model_spec.startswith("api:"):
        return OpenAIVLMBackend(model_spec.split("api:", 1)[1]), None

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    cfg = AutoConfig.from_pretrained(model_spec, trust_remote_code=True)
    if getattr(cfg, "model_type", "").startswith("qwen3_vl"):
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_spec, dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_spec, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    processor = AutoProcessor.from_pretrained(model_spec, trust_remote_code=True)
    model.eval()
    return model, processor


def resize_keep_aspect(img_path: str, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


def _pil_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode_lines(raw: str) -> list[str]:
    out = []
    for line in (raw or "").splitlines():
        s = TAG_RE.sub("", line).strip()
        if s and not s.startswith("[ERROR]"):
            out.append(s)
    return out


def _run_vlm(model, processor, prompt: str, images: list[Image.Image], max_new_tokens: int = 1536, seed: int = 0) -> str:
    if isinstance(model, OpenAIVLMBackend):
        content = [{"type": "image_url", "image_url": {"url": _pil_to_data_url(im)}} for im in images]
        content.append({"type": "text", "text": prompt})
        kw = {"temperature": 0.0} if seed == 0 else {"temperature": 0.7, "top_p": 0.8, "seed": seed}
        resp = model.client.chat.completions.create(
            model=model.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_new_tokens,
            **kw,
        )
        return (resp.choices[0].message.content or "").strip()

    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    gen_kw = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if seed > 0:
        torch.manual_seed(seed)
        gen_kw.update(do_sample=True, temperature=0.7, top_p=0.8)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def _load_figure_images(evidence: dict) -> tuple[list[Image.Image], str]:
    paths = [evidence.get("img_path", "")] + list(evidence.get("sub_images") or [])
    paths = [p for p in paths if p and Path(p).exists()]
    imgs = [resize_keep_aspect(p) for p in paths]
    labels = ["Image 1 is the full figure."] + [f"Image {i} is subfigure {i - 1}." for i in range(2, len(imgs) + 1)]
    return imgs, " ".join(labels)


def extract_sis_three_stage(evidence: dict, model, processor, max_new_tokens: int = 1536, seed: int = 0) -> list[dict]:
    caption = (evidence.get("img_caption") or "").strip()
    body = (evidence.get("img_cap_text") or "").strip()[:MAX_LOCAL_CHARS]

    p1 = PROMPT_STAGE1_TEXT.format(caption=caption, body_text=body)
    text_raw = _run_vlm(model, processor, p1, [], max_new_tokens=max_new_tokens, seed=seed)
    text_stmts = _decode_lines(text_raw)

    imgs, img_labels = _load_figure_images(evidence)
    fig_stmts, comp_stmts = [], []
    if imgs:
        p2 = PROMPT_STAGE2_VISUAL.format(image_labels=img_labels, caption=caption, body_text=body)
        fig_stmts = _decode_lines(_run_vlm(model, processor, p2, imgs, max_new_tokens=max_new_tokens, seed=seed))

        p3 = PROMPT_STAGE3_COMPLEMENT.format(
            image_labels=img_labels,
            caption=caption,
            body_text=body,
            text_facts="\n".join(text_stmts) or "(none)",
        )
        comp_stmts = _decode_lines(_run_vlm(model, processor, p3, imgs, max_new_tokens=max_new_tokens, seed=seed))

    records = [{"step": "text", "statement": s} for s in text_stmts]
    records += [{"step": "figure", "statement": s} for s in fig_stmts]
    records += [{"step": "complement", "statement": s} for s in comp_stmts]
    return records

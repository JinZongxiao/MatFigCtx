import json
import re

import torch
from PIL import Image

PANEL_TYPES = {
    "microstructure": "SEM / TEM / optical / EBSD micrograph",
    "line_plot": "x-y curve or scatter plot",
    "bar_chart": "bar chart or histogram",
    "schematic": "process schematic, diagram, or CAD model",
    "xrd": "X-ray diffraction pattern (intensity vs 2-theta)",
    "diffraction_pattern": "SAED, FFT, or 2-D electron diffraction pattern",
    "eds_map": "elemental (EDS / EDX / EPMA) map",
    "spectrum": "XPS, Raman, EDS spectrum, FTIR, or similar",
    "other": "macro photograph or anything else",
}

_LOCATE_PROMPT = (
    "This scientific figure consists of {n} panels labelled {labels}. "
    "Locate every panel. Return JSON only: a list of objects "
    '{{"label": "a", "bbox_2d": [x1, y1, x2, y2]}} with one object per panel, in label order.'
)

_CLASSIFY_PROMPT = (
    "What kind of panel is this image from a materials-science paper? Choose exactly one type:\n"
    + "\n".join(f"- {k}: {v}" for k, v in PANEL_TYPES.items())
    + "\nAnswer with the type name only."
)


def _generate(model, processor, image, prompt, max_new_tokens):
    if processor is None:
        from core.vlm_inference import _run_vlm
        return _run_vlm(model, None, prompt, [image], max_new_tokens=max_new_tokens, seed=0)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def locate_panels(model, processor, img_bgr, n: int) -> list:
    H, W = img_bgr.shape[:2]
    image = Image.fromarray(img_bgr[:, :, ::-1])
    labels = ", ".join(f"({chr(ord('a') + i)})" for i in range(n))
    text = _generate(model, processor, image, _LOCATE_PROMPT.format(n=n, labels=labels), 64 * n + 64)
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    boxes = []
    for it in items:
        b = it.get("bbox_2d") if isinstance(it, dict) else None
        if not b or len(b) != 4:
            return []
        # Qwen3-VL returns 0-1000 normalized coords
        x0, y0, x1, y1 = (int(round(v / 1000 * s)) for v, s in zip(b, (W, H, W, H)))
        x0, x1 = sorted((max(0, min(W, x0)), max(0, min(W, x1))))
        y0, y1 = sorted((max(0, min(H, y0)), max(0, min(H, y1))))
        if (x1 - x0) * (y1 - y0) < 0.01 * W * H:
            return []
        boxes.append((x0, y0, x1, y1))
    return boxes if len(boxes) >= 2 else []


def classify_panel(model, processor, image) -> str:
    text = _generate(model, processor, image, _CLASSIFY_PROMPT, 12).lower()
    for name in sorted(PANEL_TYPES, key=len, reverse=True):
        if name in text.replace(" ", "_"):
            return name
    return "other"

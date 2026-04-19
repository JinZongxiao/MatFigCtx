"""
core/vlm_inference.py

Grounded Figure Interpretation Skill: generates Scientifically Informative
Statements (SIS) from a figure image and its structured evidence chain.

Prompt structure follows the paper description:
  1. System instruction (traceability constraint)
  2. Subfigure-type prior
  3. Figure caption      [local evidence]
  4. In-text body text   [local evidence]
  5. Abstract            [global context]
  6. Conclusions         [global context]
  7. Output constraint
"""

import torch
from pathlib import Path
from PIL import Image

# Maximum characters kept per evidence field.
# Prevents prompt overflow on models with limited context windows.
_MAX_LOCAL_CHARS  = 2000   # caption + body text combined
_MAX_GLOBAL_CHARS = 1500   # abstract + conclusion combined

_SYSTEM_INSTRUCTION = (
    "You are an expert materials science assistant. "
    "Your task is to generate Scientifically Informative Statements (SIS) "
    "that describe the provided characterization figure. "
    "Every statement must be directly supported by the documentary evidence "
    "supplied below (caption, body text, abstract, or conclusions). "
    "Do not introduce claims that cannot be traced to the provided evidence. "
    "Output one statement per line. Do not add numbering, bullets, or explanations."
)

_OUTPUT_CONSTRAINT = (
    "Generate SIS for the figure above. "
    "Each statement must be an independent scientific fact "
    "(subject + verb + object), concise and precise."
)


def _build_subfig_prior(sub_image_type: list) -> str:
    """Format the subfigure-type prior block."""
    if not sub_image_type:
        return ""
    lines = []
    for i, s in enumerate(sub_image_type, 1):
        fig_type = s.get("predicted_type", s.get("type", "unknown"))
        prob     = s.get("predicted_prob", s.get("prob", 0.0))
        lines.append(f"  Panel {i}: {fig_type} (confidence: {prob:.4f})")
    return "[Subfigure Type Prior]:\n" + "\n".join(lines)


def _build_prompt(evidence: dict) -> str:
    """
    Build the structured evidence-chain prompt.

    Sections are included only when the corresponding evidence field is
    non-empty, to avoid injecting blank context blocks.
    """
    sections = [_SYSTEM_INSTRUCTION]

    # --- Subfigure-type prior ---
    prior = _build_subfig_prior(evidence.get("sub_image_type") or [])
    if prior:
        sections.append(prior)

    # --- Local evidence ---
    caption   = (evidence.get("img_caption") or "").strip()
    body_text = (evidence.get("img_cap_text") or "").strip()

    # Soft-cap local evidence to avoid overflow
    if len(body_text) > _MAX_LOCAL_CHARS:
        body_text = body_text[:_MAX_LOCAL_CHARS] + " [truncated]"

    if caption:
        sections.append(f"[Figure Caption]:\n{caption}")
    if body_text:
        sections.append(
            f"[Body Text] (paragraphs referencing this figure):\n{body_text}"
        )

    # --- Global context ---
    abstract   = (evidence.get("abstract") or "").strip()
    conclusion = (evidence.get("conclusion") or "").strip()

    if len(abstract) > _MAX_GLOBAL_CHARS:
        abstract = abstract[:_MAX_GLOBAL_CHARS] + " [truncated]"
    if len(conclusion) > _MAX_GLOBAL_CHARS:
        conclusion = conclusion[:_MAX_GLOBAL_CHARS] + " [truncated]"

    if abstract:
        sections.append(f"[Abstract]:\n{abstract}")
    if conclusion:
        sections.append(f"[Conclusions]:\n{conclusion}")

    sections.append(_OUTPUT_CONSTRAINT)
    return "\n\n".join(sections)


def generate_sis(evidence: dict,
                 model,
                 processor,
                 max_new_tokens: int = 1024) -> str:
    """
    Generate SIS for a single figure evidence record.

    Args:
        evidence: dict with keys img_path, img_caption, img_cap_text,
                  abstract, conclusion, sub_images, sub_image_type.
        model: loaded VLM (Qwen3-VL or InternVL).
        processor: corresponding AutoProcessor.
        max_new_tokens: generation budget.

    Returns:
        Generated SIS as a newline-separated string.
    """
    img_path = evidence.get("img_path", "")
    sub_imgs = evidence.get("sub_images") or []

    # Composite figure image first, then individual panels
    image_paths = [img_path] + list(sub_imgs)
    image_paths = [p for p in image_paths if p and Path(p).exists()]

    content = []
    for p in image_paths:
        content.append({"type": "image", "image": Image.open(p).convert("RGB")})
    content.append({"type": "text", "text": _build_prompt(evidence)})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = generated_ids[0][inputs.input_ids.shape[-1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def load_vlm(model_path: str):
    """
    Load a VLM and its processor from a local model directory.

    Tries Qwen3VLForConditionalGeneration first; falls back to the generic
    AutoModelForMultimodalLM for other HuggingFace-compatible VLMs.
    """
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    except (ImportError, OSError):
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor

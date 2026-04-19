"""
benchmark/_vlm_utils.py
Shared utilities for loading Qwen3-VL / InternVL3.5 and running inference.
Used by ablation.py, matcha_pipeline.py, matcha_eval.py.
"""

import os
from pathlib import Path

import torch
from PIL import Image, ImageOps


# -------- Image preprocessing --------
def resize_image(image_path: str, new_size: int = 512, cache_root: str = None) -> str:
    img = Image.open(image_path)
    w, h = img.size
    max_side = max(w, h)
    dw, dh = max_side - w, max_side - h
    padding = (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2)
    img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    img = img.resize((new_size, new_size), Image.LANCZOS)
    filename = os.path.basename(image_path)
    if cache_root:
        os.makedirs(cache_root, exist_ok=True)
        out_path = os.path.join(cache_root, filename)
    else:
        out_path = image_path
    img.save(out_path)
    return out_path


# -------- Model loader --------
def load_vlm_model(model_key: str, model_path: str):
    """Return (model, processor) for Qwen3-VL or InternVL3.5."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    if model_key.startswith("qwen"):
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
    print(f"[INFO] Loaded {model_key} from {model_path}")
    return model, processor


# -------- Inference --------
def _run_qwen3(model, processor, img_paths, prompt, max_new_tokens):
    content = [{"type": "image", "image": p} for p in img_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _run_internvl(model, processor, img_paths, prompt, max_new_tokens):
    pil_images = [Image.open(p).convert("RGB") for p in img_paths]
    if not pil_images:
        hf_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, return_tensors="pt")
    else:
        hf_content = [{"type": "image"} for _ in pil_images]
        hf_content.append({"type": "text", "text": prompt})
        hf_messages = [{"role": "user", "content": hf_content}]
        text = processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=pil_images, return_tensors="pt")

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def run_vlm(model_key: str, model, processor, img_paths, prompt,
            max_new_tokens: int = 1024):
    if model_key.startswith("qwen"):
        return _run_qwen3(model, processor, img_paths, prompt, max_new_tokens)
    return _run_internvl(model, processor, img_paths, prompt, max_new_tokens)

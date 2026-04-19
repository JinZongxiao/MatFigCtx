"""
core/subfig_classifier.py
子图分类模块：ResNet50三分类（microstructure / xrd / line_plot）
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["line_plot", "xrd", "microstructure"]

_model_cache = {}


def _get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_model(num_classes: int):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(model_path: str):
    if model_path in _model_cache:
        return _model_cache[model_path]

    ckpt        = torch.load(model_path, map_location=DEVICE)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    model       = _build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    _model_cache[model_path] = (model, class_names)
    return model, class_names


@torch.no_grad()
def classify_image(img_path: str, model_path: str) -> dict:
    """
    分类单张图像
    返回: {"type": "microstructure", "prob": 0.95}
    """
    model, class_names = load_classifier(model_path)
    transform = _get_transforms()

    img    = Image.open(img_path).convert("RGB")
    x      = transform(img).unsqueeze(0).to(DEVICE)
    prob   = torch.softmax(model(x), dim=1)[0]
    idx    = prob.argmax().item()

    return {
        "type": class_names[idx],
        "prob": round(prob[idx].item(), 4),
    }


def classify_subfigs(subfig_paths: list, model_path: str) -> list:
    """批量分类子图，返回分类结果列表"""
    results = []
    for path in subfig_paths:
        try:
            result = classify_image(path, model_path)
            result["path"] = path
        except Exception as e:
            result = {"path": path, "type": "unknown", "prob": 0.0, "error": str(e)}
        results.append(result)
    return results

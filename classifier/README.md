# Subfigure Classifier Checkpoint

This directory holds the ResNet-50 checkpoint for the three-class subfigure classifier
(microstructural image / XRD pattern / property curve).

## Files

| File | Format | Use |
|------|--------|-----|
| `image_classifier.pt` | `torch.save({"model_state_dict": ..., "class_names": [...]})` | **Use this one** — loaded by `core/subfig_classifier.py` |
| `best_model.pth` | raw `model.state_dict()` | intermediate training artifact — not required for inference |

**Only `image_classifier.pt` is needed to run the framework.**

## Download

Because both files exceed GitHub's 50 MB file-size limit, they are hosted externally.

> **Download `image_classifier.pt`:**  
> [JinZONGXIAO/MatFigCtx_classifier on Hugging Face](https://huggingface.co/JinZONGXIAO/MatFigCtx_classifier)

Place the downloaded file at the path configured in `configs/config.yaml`:

```yaml
CLASSIFIER_PATH: "/path/to/classifier/image_classifier.pt"
```

## Classes

```python
CLASS_NAMES = ["line_plot", "xrd", "microstructure"]
```

## Quick test

```python
from core.subfig_classifier import classify_image
result = classify_image("path/to/panel.jpg", "classifier/image_classifier.pt")
# {"type": "microstructure", "prob": 0.97}
```

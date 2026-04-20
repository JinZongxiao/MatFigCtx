# 子图分类器检查点

[English](README.md) | [中文](README.zh-CN.md)

本目录用于存放 MatFigCtx 使用的三分类 ResNet-50 子图分类器权重（显微组织图 / XRD 图谱 / 性能曲线）。

## 文件说明

| 文件 | 格式 | 用途 |
|------|------|------|
| `image_classifier.pt` | `torch.save({"model_state_dict": ..., "class_names": [...]})` | **正式推理使用文件**（由 `core/subfig_classifier.py` 加载） |
| `best_model.pth` | 原始 `model.state_dict()` | 训练过程中的中间产物，推理阶段非必需 |

仅需 `image_classifier.pt` 即可运行框架。

## 下载

由于文件体积超过 GitHub 50MB 限制，模型托管在 Hugging Face：

> **下载地址：**  
> [JinZONGXIAO/MatFigCtx_classifier](https://huggingface.co/JinZONGXIAO/MatFigCtx_classifier)

下载后在 `configs/config.yaml` 中配置路径：

```yaml
CLASSIFIER_PATH: "/path/to/classifier/image_classifier.pt"
```

## 类别标签

```python
CLASS_NAMES = ["line_plot", "xrd", "microstructure"]
```

## 快速测试

```python
from core.subfig_classifier import classify_image
result = classify_image("path/to/panel.jpg", "classifier/image_classifier.pt")
# {"type": "microstructure", "prob": 0.97}
```

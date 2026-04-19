"""
core/subfig_segmentor.py
子图分割模块：基于投影法切割复合图像
"""

import cv2
import numpy as np
import os
import re
from pathlib import Path


def _get_intervals(projection, min_len=10):
    intervals, start = [], None
    for i, val in enumerate(projection):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start >= min_len:
                intervals.append((start, i))
            start = None
    if start is not None and len(projection) - start >= min_len:
        intervals.append((start, len(projection)))
    return intervals


def segment_image(img_path: str, output_dir: str) -> list:
    """
    分割单张图像，返回子图路径列表
    """
    img_name = Path(img_path).stem
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        return []

    H, W = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    row_proj      = np.sum(binary, axis=1)
    row_intervals = _get_intervals(row_proj, min_len=int(H * 0.1))

    panels = []
    for y1, y2 in row_intervals:
        col_proj      = np.sum(binary[y1:y2, :], axis=0)
        col_intervals = _get_intervals(col_proj, min_len=int(W * 0.1))
        for x1, x2 in col_intervals:
            pad = 2
            roi = img[max(0, y1-pad):min(H, y2+pad),
                      max(0, x1-pad):min(W, x2+pad)]
            panels.append((y1, x1, roi))

    panels.sort(key=lambda p: (int(p[0] / (H * 0.1)), p[1]))

    saved = []
    for i, (_, _, sub_img) in enumerate(panels, 1):
        out_path = os.path.join(output_dir, f"{img_name}_{i}.jpg")
        cv2.imwrite(out_path, sub_img)
        saved.append(out_path)

    return saved


def segment_all_images(image_dir: str) -> dict:
    """批量分割目录下所有图像，返回 {原图路径: [子图路径列表]}"""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    result = {}
    for fname in os.listdir(image_dir):
        if Path(fname).suffix.lower() not in valid_ext:
            continue
        # 跳过已分割的子图（文件名含 _数字）
        if re.search(r'_\d+$', Path(fname).stem):
            continue
        img_path  = os.path.join(image_dir, fname)
        sub_imgs  = segment_image(img_path, image_dir)
        result[img_path] = sub_imgs
    return result

import os
import re
from pathlib import Path

import cv2
import numpy as np

WHITE_LEVEL = 235
GAP_NOISE = 0.002
MIN_GAP_FRAC = 0.008
MIN_GAP_PX = 3
MIN_AREA_FRAC = 0.04
MIN_SIDE_FRAC = 0.12
MAX_DEPTH = 4
PHOTO_FILL = 0.5   # only run seam detection on dense micrographs so plot axes aren't split
SEAM_RATIO = 4.0
SEAM_MIN_STEP = 25
SEAM_MARGIN = 0.12
LABEL_FRAC = 0.01
OVERLAP_TOL = 0.05


def _trim(mask, box):
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    rows, cols = np.where(sub.any(axis=1))[0], np.where(sub.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return (x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1)


def _gaps(profile, length):
    is_gap = profile <= max(1, GAP_NOISE * length)
    min_gap = max(MIN_GAP_PX, int(MIN_GAP_FRAC * len(profile)))
    gaps, start = [], None
    for i, g in enumerate(is_gap):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= min_gap and start > 0:
                gaps.append((start, i))
            start = None
    return gaps


def _seams(gray, axis):
    d = np.abs(np.diff(gray.astype(np.int16), axis=axis)).mean(axis=1 - axis)
    n = len(d) + 1
    base = np.median(d)
    cand = [
        i for i in np.argsort(d)[::-1]
        if d[i] > max(SEAM_RATIO * base, SEAM_MIN_STEP)
        and SEAM_MARGIN * n < i < (1 - SEAM_MARGIN) * n
    ]
    picked = []
    for i in cand:
        # keep seams apart so both edges of a black divider line aren't picked
        if all(abs(i - j) >= MIN_SIDE_FRAC * n for j in picked):
            picked.append(i)
    return [(i + 1, i + 2) for i in sorted(picked)]


def _xy_cut(mask, box, depth=0, gray=None):
    box = _trim(mask, box)
    if box is None:
        return []
    if depth >= MAX_DEPTH:
        return [box]
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    row_gaps = _gaps(sub.sum(axis=1), x1 - x0)
    col_gaps = _gaps(sub.sum(axis=0), y1 - y0)
    if not row_gaps and not col_gaps and gray is not None and sub.mean() > PHOTO_FILL:
        g = gray[y0:y1, x0:x1]
        row_gaps, col_gaps = _seams(g, 0), _seams(g, 1)
    if not row_gaps and not col_gaps:
        return [box]
    widest = lambda gaps, n: max((e - s) / n for s, e in gaps) if gaps else 0
    by_rows = widest(row_gaps, y1 - y0) >= widest(col_gaps, x1 - x0)
    cuts = row_gaps if by_rows else col_gaps
    edges = [0] + [(s + e) // 2 for s, e in cuts] + [(y1 - y0) if by_rows else (x1 - x0)]
    leaves = []
    for a, b in zip(edges[:-1], edges[1:]):
        child = (x0, y0 + a, x1, y0 + b) if by_rows else (x0 + a, y0, x0 + b, y1)
        leaves += _xy_cut(mask, child, depth + 1, gray)
    return leaves


def _is_fragment(box, W, H):
    w, h = box[2] - box[0], box[3] - box[1]
    return w * h < MIN_AREA_FRAC * W * H or w < MIN_SIDE_FRAC * W or h < MIN_SIDE_FRAC * H


def _overlaps(a, b, tol=OVERLAP_TOL):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return False
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h > tol * smaller


def _merge_fragments(boxes, W, H):
    keep = [b for b in boxes if not _is_fragment(b, W, H)]
    if not keep:
        return boxes

    def gap(a, b):
        dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return dx + dy

    def corner(frag, b):
        # panel letters usually sit at the top-left of their own panel
        dx, dy = b[0] - frag[0], b[1] - frag[1]
        return abs(dx) + abs(dy) + (W + H) * (dx < -0.02 * W or dy < -0.02 * H)

    for frag in (b for b in boxes if _is_fragment(b, W, H)):
        small = (frag[2] - frag[0]) * (frag[3] - frag[1]) < LABEL_FRAC * W * H
        k = min(range(len(keep)), key=lambda i: corner(frag, keep[i]) if small else gap(frag, keep[i]))
        a = keep[k]
        merged = (min(a[0], frag[0]), min(a[1], frag[1]), max(a[2], frag[2]), max(a[3], frag[3]))
        # drop shared legends/axis titles rather than letting a merge swallow a neighbor
        if not any(_overlaps(merged, b) for j, b in enumerate(keep) if j != k):
            keep[k] = merged
    return keep


def _valid(boxes) -> bool:
    return not any(_overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def caption_panel_count(caption: str) -> int:
    c = caption or ""
    found = set(re.findall(r"\(([a-l])\)", c))
    for a, b in (
        re.findall(r"\(([a-l])\)\s*[–-]\s*\(([a-l])\)", c)
        + re.findall(r"\(([a-l])\s*[–-]\s*([a-l])\)", c)
    ):
        found |= {chr(x) for x in range(ord(a), ord(b) + 1)}
    for group in re.findall(r"\(([a-l](?:\s*(?:,|and|&)\s*[a-l])+)\)", c):
        found |= set(re.findall(r"[a-l]", group))
    return len(found)


def find_panels(img, seams: bool = False) -> list:
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray < WHITE_LEVEL
    boxes = _merge_fragments(_xy_cut(mask, (0, 0, W, H), gray=gray if seams else None), W, H)
    if len(boxes) <= 1:
        return []
    boxes.sort(key=lambda b: (b[1], b[0]))
    rows, order = [], []
    for b in boxes:
        if rows and b[1] < rows[-1][0][1] + 0.5 * (rows[-1][0][3] - rows[-1][0][1]):
            rows[-1].append(b)
        else:
            rows.append([b])
    for r in rows:
        order += sorted(r, key=lambda b: b[0])
    return order


def find_panels_guided(img, caption: str = "", locator=None):
    expected = caption_panel_count(caption)
    plain = find_panels(img)
    if expected < 2:
        return (plain, "xycut") if _valid(plain) else ([], "xycut_fallback")
    if len(plain) == expected and _valid(plain):
        return plain, "xycut"
    seamed = find_panels(img, seams=True)
    if len(seamed) == expected and _valid(seamed):
        return seamed, "xycut+seams"
    if locator is not None:
        located = locator(img, expected)
        if located and _valid(located):
            return located, "vlm"
    # better to pass the whole figure unsplit than feed overlapping bad crops
    return [], "unsplit"


def segment_image(img_path: str, output_dir: str, caption: str = "", locator=None) -> list:
    img = cv2.imread(img_path)
    if img is None:
        return []
    os.makedirs(output_dir, exist_ok=True)
    H, W = img.shape[:2]
    stem = Path(img_path).stem
    boxes, _ = find_panels_guided(img, caption, locator)
    saved = []
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        pad = 2
        roi = img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
        out = os.path.join(output_dir, f"{stem}_p{i}.jpg")
        cv2.imwrite(out, roi)
        saved.append(out)
    return saved

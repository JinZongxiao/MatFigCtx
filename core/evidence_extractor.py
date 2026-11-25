import json
import os
import re
from pathlib import Path

import cv2
import markdown
from bs4 import BeautifulSoup

_FIG_WORD = re.compile(r"\bfig(?:ure)?s?\.?\s*", re.IGNORECASE)
_REF_TAIL = re.compile(
    r"(?:\d+\s*(?:\(?\s*[a-z](?![a-z])(?:\s*[–\-,]\s*[a-z](?![a-z]))*\s*\)?)?"
    r"(?:\s*(?:,|;|and|&|to|–|-|—)\s*)?)+",
    re.IGNORECASE,
)
_NOT_MAIN_FIG = re.compile(r"(?:supplementary|supporting|suppl\.?|extended data)\s+$", re.IGNORECASE)
_AFTER_CONCLUSION = (
    "acknowledg", "reference", "declaration", "funding", "data availability",
    "credit", "appendix", "supplementary", "conflict of interest",
)


def _extract_fig_number(caption: str):
    if not isinstance(caption, str):
        return None
    m = re.search(r"\bfig(?:ure)?s?\s*[.:\-–]*\s*(\d+)\b", caption, re.IGNORECASE)
    if m:
        return m.group(1)
    # MinerU occasionally clips the start of "Fig.": "ig. 9.", ". 1 |"
    m = re.match(r"^\W*(?:ig(?:ure)?s?\s*\.?\s*|\.\s*)(\d+)\s*[.:|]", caption, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^\s*(\d{1,2})\s+(?=\(?[A-Za-z])", caption)
    return m.group(1) if m else None


def fig_refs(text: str) -> set:
    found = set()
    for m in _FIG_WORD.finditer(text or ""):
        tail = _REF_TAIL.match(text, m.end())
        # skip Supplementary / Extended Data figures
        if not tail or _NOT_MAIN_FIG.search(text[max(0, m.start() - 25):m.start()]):
            continue
        span = re.sub(
            r"\(?\s*(?<![a-z])[a-z](?![a-z])(?:\s*[–\-,]\s*[a-z](?![a-z]))*\s*\)?",
            " ", tail.group(0), flags=re.I,
        )
        nums = [(int(a), int(b) if b else None) for a, b in re.findall(r"(\d+)(?:\s*(?:–|-|—|to)\s*(\d+))?", span)]
        for a, b in nums:
            found.update(range(a, b + 1) if b and 0 < b - a < 15 else [a])
    return found


def figure_text(paragraphs: list, number: int) -> str:
    return "\n".join(p for p in paragraphs if number in fig_refs(p))


def _parse_abstract_conclusion(md_path: str) -> dict:
    with open(md_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(markdown.markdown(f.read()), "html.parser")

    abstract, conclusion = "", ""
    for title in soup.find_all(["h1", "h2", "h3"]):
        lo_title = title.text.lower()
        if not abstract and any(k in lo_title for k in ("abstract", "a b s t r a c t")) and "graphical" not in lo_title:
            for sib in title.next_siblings:
                if sib.name == "p" and len(sib.text) > 200:
                    abstract = sib.text.strip()
                    break

        if any(k in lo_title for k in ("conclusion", "conclusions")):
            parts = []
            for sib in title.next_siblings:
                name = getattr(sib, "name", None)
                if name in ("h1", "h2", "h3"):
                    break
                if name == "p":
                    texts = [sib.get_text(strip=True)]
                elif name in ("ol", "ul"):
                    texts = [li.get_text(strip=True) for li in sib.find_all("li")]
                else:
                    continue
                for t in texts:
                    lo = t.lower()
                    if lo.startswith(("fig", "[", "the authors acknowledge")) or lo.startswith(_AFTER_CONCLUSION):
                        continue
                    parts.append(t)
            if parts:
                conclusion = "\n".join(parts).strip()

    return {"abstract": abstract, "conclusion": conclusion}


def _subfigures(img_path: str, caption: str, out_dir: str, vlm=None, classify: bool = False):
    from core.subfig_segmentor import find_panels_guided

    img = cv2.imread(img_path)
    if img is None:
        return [], []
    locator = None
    if vlm is not None:
        from core.panel_vlm import locate_panels
        locator = lambda im, n: locate_panels(*vlm, im, n)

    boxes, _ = find_panels_guided(img, caption, locator)
    stem, crops = Path(img_path).stem, []
    for k, (x0, y0, x1, y1) in enumerate(boxes, 1):
        out = os.path.join(out_dir, f"{stem}_p{k}.jpg")
        cv2.imwrite(out, img[y0:y1, x0:x1])
        crops.append(out)

    types = []
    if classify and vlm is not None:
        from PIL import Image
        from core.panel_vlm import classify_panel
        targets = crops or [img_path]
        types = [{"type": classify_panel(*vlm, Image.open(p).convert("RGB")), "path": p} for p in targets]
    return crops, types


def extract_evidence_chain(
    content_list_path: str,
    md_path: str,
    vlm=None,
    crops_dir: str = None,
    classify_panels: bool = False,
) -> list:
    with open(content_list_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    title = ""
    for item in raw[:3]:
        if item.get("type") == "text" and len(item.get("text", "")) > 10:
            title = item["text"]
            break

    ac = _parse_abstract_conclusion(md_path)
    base_dir = str(Path(content_list_path).parent.parent)
    paragraphs = [x["text"] for x in raw if x.get("type") == "text" and isinstance(x.get("text"), str)]

    results = []
    for item in raw:
        if item.get("type") != "image":
            continue
        img_path = os.path.join(base_dir, item.get("img_path", ""))
        if not os.path.isfile(img_path):
            continue
        caps = item.get("img_caption") or []
        caption = caps[0] if caps else ""
        if not caption:
            continue

        num = _extract_fig_number(caption)
        cap_text = figure_text(paragraphs, int(num)) if num else ""

        out_dir = crops_dir or str(Path(img_path).parent)
        os.makedirs(out_dir, exist_ok=True)
        sub_imgs, sub_types = _subfigures(img_path, caption, out_dir, vlm=vlm, classify=classify_panels)

        results.append({
            "img_path": img_path,
            "img_caption": caption,
            "img_cap_text": cap_text.strip(),
            "literature_title": title,
            "abstract": ac["abstract"],
            "conclusion": ac["conclusion"],
            "sub_images": sub_imgs,
            "sub_image_type": sub_types,
        })
    return results


def extract_all(paper_dir: str, ict_output_dir: str, vlm=None, crops_root: str = None, classify_panels: bool = False):
    os.makedirs(ict_output_dir, exist_ok=True)
    for subdir in sorted(Path(paper_dir).iterdir()):
        if not subdir.is_dir():
            continue
        md_dir = subdir / "markdowns"
        cl_path = md_dir / f"{subdir.name}_content_list.json"
        md_path = md_dir / f"{subdir.name}.md"
        if not cl_path.exists() or not md_path.exists():
            continue
        out_path = Path(ict_output_dir) / f"{subdir.name}_image_captions_with_texts.json"
        if out_path.exists():
            continue
        try:
            crops = str(Path(crops_root) / subdir.name) if crops_root else None
            ev = extract_evidence_chain(str(cl_path), str(md_path), vlm=vlm, crops_dir=crops, classify_panels=classify_panels)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False, indent=2)
            print(f"{subdir.name}: {len(ev)} figures")
        except Exception as e:
            print(f"[ERR] {subdir.name}: {e}")

"""
core/evidence_extractor.py
证据链抽取模块：从论文JSON+MD中提取图文证据链
"""

import json
import os
import re
from pathlib import Path
import markdown
from bs4 import BeautifulSoup


def _extract_fig_number(caption: str):
    if not isinstance(caption, str):
        return None
    m = re.search(r'\bfigs?(?:ure)?\s*\.?\s*(\d+)\b', caption, re.IGNORECASE)
    return m.group(1) if m else None


def _parse_abstract_conclusion(md_path: str) -> dict:
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")

    abstract   = ""
    conclusion = ""

    for title in soup.find_all(["h1", "h2", "h3"]):
        text_lower = title.text.lower()

        if any(kw in text_lower for kw in ["abstract", "a b s t r a c t", "摘要"]):
            for sib in title.next_siblings:
                if sib.name == "p" and len(sib.text) > 200:
                    abstract = sib.text.strip()
                    break

        if any(kw in text_lower for kw in ["conclusion", "conclusions", "结论"]):
            parts = []
            for sib in title.next_siblings:
                name = getattr(sib, "name", None)
                if name == "p":
                    texts = [sib.get_text(strip=True)]
                elif name in ["ol", "ul"]:
                    texts = [li.get_text(strip=True) for li in sib.find_all("li")]
                else:
                    continue
                for t in texts:
                    lo = t.lower()
                    if (lo.startswith("fig") or lo.startswith("[") or
                            lo.startswith("the authors acknowledge")):
                        continue
                    parts.append(t)
            conclusion = "\n".join(parts).strip()

    return {"abstract": abstract, "conclusion": conclusion}


def extract_evidence_chain(content_list_path: str,
                           md_path: str,
                           classifier_model_path: str = None) -> list:
    """
    从单篇论文提取证据链
    返回: [{img_path, img_caption, img_cap_text, literature_title,
             abstract, conclusion, sub_images, sub_image_type}, ...]
    """
    from core.subfig_segmentor import segment_image
    from core.subfig_classifier import classify_subfigs

    with open(content_list_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 提取论文标题
    literature_title = ""
    for item in raw[:3]:
        if item.get("type") == "text" and len(item.get("text", "")) > 10:
            literature_title = item["text"]
            break

    # 提取摘要+结论
    ac = _parse_abstract_conclusion(md_path)

    # 提取图像路径和图注
    img_paths, img_captions = [], []
    base_dir = str(Path(content_list_path).parent.parent)

    for item in raw:
        if item["type"] == "image":
            img_path = os.path.join(base_dir, item.get("img_path", ""))
            if os.path.exists(img_path):
                img_paths.append(img_path)
                try:
                    img_captions.append(item["img_caption"][0])
                except Exception:
                    img_captions.append("")

    # 提取图片对应的正文段落
    results = []
    for i, (img_path, caption) in enumerate(zip(img_paths, img_captions)):
        if not caption:
            continue

        number   = _extract_fig_number(caption)
        cap_text = ""

        if number:
            pattern = re.compile(
                rf'(?:\bfigs?(?:ure)?\.?\s*{re.escape(number)}(?!\d)\s*'
                rf'(?:\(?\s*[a-z]\s*\)?(?:\s*(?:,|;|and|&|–|-)\s*\(?\s*[a-z]\s*\)?)*)?)',
                re.IGNORECASE
            )
            for item in raw:
                txt = item.get("text", "")
                if item.get("type") == "text" and isinstance(txt, str) and pattern.search(txt):
                    cap_text += txt + "\n"

        # 子图分割
        img_dir  = str(Path(img_path).parent)
        img_stem = Path(img_path).stem
        sub_imgs = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if re.match(rf'^{re.escape(img_stem)}_\d+\.(jpg|jpeg|png)$', f, re.IGNORECASE)
        ]

        # 如果没有预分割，执行分割
        if not sub_imgs:
            sub_imgs = segment_image(img_path, img_dir)

        # 子图分类
        sub_image_type = []
        if classifier_model_path and sub_imgs:
            sub_image_type = classify_subfigs(sub_imgs, classifier_model_path)

        results.append({
            "img_path":        img_path,
            "img_caption":     caption,
            "img_cap_text":    cap_text.strip(),
            "literature_title": literature_title,
            "abstract":        ac["abstract"],
            "conclusion":      ac["conclusion"],
            "sub_images":      sub_imgs,
            "sub_image_type":  sub_image_type,
        })

    return results


def extract_all(paper_dir: str, ict_output_dir: str,
                classifier_model_path: str = None):
    """批量处理所有论文"""
    os.makedirs(ict_output_dir, exist_ok=True)
    paper_dir = Path(paper_dir)

    for subdir in paper_dir.iterdir():
        if not subdir.is_dir():
            continue
        md_dir = subdir / "markdowns"
        cl_path = md_dir / f"{subdir.name}_content_list.json"
        md_path = md_dir / f"{subdir.name}.md"

        if not cl_path.exists() or not md_path.exists():
            continue

        out_path = Path(ict_output_dir) / f"{subdir.name}_image_captions_with_texts.json"
        if out_path.exists():
            print(f"[SKIP] {subdir.name}")
            continue

        print(f"[Extracting] {subdir.name}")
        try:
            evidence = extract_evidence_chain(
                str(cl_path), str(md_path), classifier_model_path
            )
            with open(str(out_path), "w", encoding="utf-8") as f:
                json.dump(evidence, f, ensure_ascii=False, indent=2)
            print(f"  → {len(evidence)} figures extracted")
        except Exception as e:
            print(f"[ERROR] {subdir.name}: {e}")

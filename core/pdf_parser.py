"""
core/pdf_parser.py
PDF解析模块：基于MinerU将PDF转为结构化Markdown和图像
"""

import os
from pathlib import Path


def parse_pdf(pdf_path: str, output_base_dir: str) -> dict:
    """
    解析单个PDF文件
    返回: {"md_path": ..., "image_dir": ..., "content_list_path": ...}
    """
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    pdf_path    = Path(pdf_path)
    folder_name = pdf_path.stem
    output_dir  = Path(output_base_dir) / folder_name
    image_dir   = output_dir / "images"
    md_dir      = output_dir / "markdowns"
    image_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    image_writer = FileBasedDataWriter(str(image_dir))
    md_writer    = FileBasedDataWriter(str(md_dir))
    reader       = FileBasedDataReader("")
    pdf_bytes    = reader.read(str(pdf_path))
    ds           = PymuDocDataset(pdf_bytes)

    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result  = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result  = infer_result.pipe_txt_mode(image_writer)

    md_filename = f"{folder_name}.md"
    cl_filename = f"{folder_name}_content_list.json"
    pipe_result.dump_md(md_writer, md_filename, "images")
    pipe_result.dump_content_list(md_writer, cl_filename, "images")

    return {
        "md_path":           str(md_dir / md_filename),
        "image_dir":         str(image_dir),
        "content_list_path": str(md_dir / cl_filename),
    }


def parse_all_pdfs(pdf_dir: str, output_base_dir: str) -> list:
    """批量解析目录下所有PDF"""
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF files")
    results = []
    for pdf_file in pdf_files:
        print(f"[Parsing] {pdf_file.name}")
        try:
            result = parse_pdf(str(pdf_file), output_base_dir)
            result["source"] = str(pdf_file)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {pdf_file.name}: {e}")
    return results

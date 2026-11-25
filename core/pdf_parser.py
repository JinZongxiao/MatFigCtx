from pathlib import Path


def parse_pdf(pdf_path: str, output_base_dir: str) -> dict:
    from magic_pdf.config.enums import SupportedPdfParseMethod
    from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    out_dir = Path(output_base_dir) / stem
    img_dir = out_dir / "images"
    md_dir = out_dir / "markdowns"
    img_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    img_writer = FileBasedDataWriter(str(img_dir))
    md_writer = FileBasedDataWriter(str(md_dir))
    ds = PymuDocDataset(FileBasedDataReader("").read(str(pdf_path)))

    if ds.classify() == SupportedPdfParseMethod.OCR:
        pipe_res = ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(img_writer)
    else:
        pipe_res = ds.apply(doc_analyze, ocr=False).pipe_txt_mode(img_writer)

    md_name = f"{stem}.md"
    cl_name = f"{stem}_content_list.json"
    pipe_res.dump_md(md_writer, md_name, "images")
    pipe_res.dump_content_list(md_writer, cl_name, "images")

    return {
        "md_path": str(md_dir / md_name),
        "image_dir": str(img_dir),
        "content_list_path": str(md_dir / cl_name),
    }


def parse_all_pdfs(pdf_dir: str, output_base_dir: str) -> list:
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")
    results = []
    for p in pdfs:
        print(f"Parsing {p.name}")
        try:
            res = parse_pdf(str(p), output_base_dir)
            res["source"] = str(p)
            results.append(res)
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
    return results

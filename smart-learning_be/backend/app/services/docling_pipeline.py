from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions


def convert_pdf_to_markdown(
    pdf_path: str,
    ocr_mode: str = "auto",   # "off", "auto", "force"
    ocr_langs: List[str] = ["vie", "eng"],  # song ngữ Việt - Anh
    do_tables: bool = True
) -> str:
    p = PdfPipelineOptions()
    p.do_table_structure = bool(do_tables)
    p.table_structure_options.do_cell_matching = bool(do_tables)

    if ocr_mode != "off":
        p.do_ocr = True
        # Truyền list[str] thay vì string
        ocr_opts = TesseractCliOcrOptions(lang=ocr_langs)

        if ocr_mode == "force":
            ocr_opts.force_full_page_ocr = True

        p.ocr_options = ocr_opts
    else:
        p.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=p)}
    )
    doc = converter.convert(Path(pdf_path)).document
    return doc.export_to_markdown()

"""Runs Docling once per document and hands out two views of the result:
labeled text items (used for heading-aware chunking) and tables (used for
table extraction). One conversion, cached by file hash, shared by both --
no reason to parse the same document twice with two different tools.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from sec_rag.cache import cached_call


@dataclass(frozen=True)
class DoclingTextItem:
    label: str  # "section_header", "text", "list_item", "page_header", "page_footer", ...
    page_no: int
    text: str


@dataclass(frozen=True)
class DoclingTable:
    table_index: int
    page_no: int
    markdown: str


@dataclass(frozen=True)
class DoclingExtraction:
    text_items: list[DoclingTextItem]
    tables: list[DoclingTable]


def _convert_uncached(pdf_path: Path) -> dict:
    opts = PdfPipelineOptions()
    opts.do_ocr = False  # these filings have a real text layer; OCR is unnecessary and slow
    opts.do_table_structure = True
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_chart_extraction = False

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    result = converter.convert(str(pdf_path))
    doc = result.document

    text_items = []
    for t in doc.texts:
        if not t.text.strip() or not t.prov:
            continue
        text_items.append({"label": str(t.label), "page_no": t.prov[0].page_no, "text": t.text})

    tables = []
    for i, t in enumerate(doc.tables):
        pages = sorted(set(p.page_no for p in t.prov))
        for page_no in pages:
            tables.append({"table_index": i, "page_no": page_no, "markdown": t.export_to_markdown(doc)})

    return {"text_items": text_items, "tables": tables}


def extract_docling(pdf_path: Path) -> DoclingExtraction:
    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    raw = cached_call("docling_extraction", file_hash, lambda: _convert_uncached(pdf_path))
    return DoclingExtraction(
        text_items=[DoclingTextItem(**t) for t in raw["text_items"]],
        tables=[DoclingTable(**t) for t in raw["tables"]],
    )

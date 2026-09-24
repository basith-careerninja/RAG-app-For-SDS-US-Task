"""Table extraction: Docling for structure, Gemini for an independent
cross-check, plus a short summary of each table for embedding.

Each table ends up stored two ways: a short LLM summary (embedded for
search) and the full markdown table (what actually gets passed to the
generator once that summary is retrieved).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from google import genai
from google.genai import types

from sec_rag.cache import cached_call
from sec_rag.config import get_settings
from sec_rag.numbers import extract_numbers
from sec_rag.openai_utils import call_with_retry_gemini

GEMINI_MODEL = "gemini-3.1-flash-lite"

_TRANSCRIBE_PROMPT = """This is a page from an SEC 10-Q filing. Transcribe every table on this page into clean GitHub-flavored markdown tables. Preserve exact numbers, signs (parentheses for negatives), $ signs, and column headers (including period labels like "Three Months Ended..."). If there is no table on this page, respond with "NO_TABLE". Do not add commentary -- output only the markdown table(s)."""

_SUMMARY_PROMPT = """Write a one-to-two sentence summary of this table for a search index: name the financial statement or note it's from, the line items it covers, and the periods/dates covered. Be specific and factual, no commentary.

Table (markdown):
{markdown}
"""


@dataclass(frozen=True)
class DoclingTable:
    table_index: int
    page_no: int
    markdown: str


@dataclass(frozen=True)
class TableRecord:
    page_no: int
    docling_markdown: str
    gemini_markdown: str
    agreement: float  # fraction of docling's numbers also found in gemini's transcription
    summary: str


def _extract_docling_tables_uncached(pdf_path: Path) -> list[dict]:
    opts = PdfPipelineOptions()
    opts.do_ocr = False  # this filing has a real text layer; OCR is unnecessary and slow
    opts.do_table_structure = True
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_chart_extraction = False

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    result = converter.convert(str(pdf_path))
    doc = result.document

    tables = []
    for i, t in enumerate(doc.tables):
        pages = sorted(set(p.page_no for p in t.prov))
        for page_no in pages:
            tables.append({"table_index": i, "page_no": page_no, "markdown": t.export_to_markdown(doc)})
    return tables


def extract_docling_tables(pdf_path: Path) -> list[DoclingTable]:
    # Docling's layout model is slow; cache by file hash so it only runs once per PDF.
    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    raw = cached_call("docling_tables", file_hash, lambda: _extract_docling_tables_uncached(pdf_path))
    return [DoclingTable(**t) for t in raw]


_gemini_client_instance: genai.Client | None = None


def _gemini_client() -> genai.Client:
    # a fresh Client() per call can get garbage-collected mid-request; reuse one
    global _gemini_client_instance
    if _gemini_client_instance is None:
        _gemini_client_instance = genai.Client(api_key=get_settings().gemini_api_key)
    return _gemini_client_instance


def render_page_png(pdf_path: Path, page_number: int, dpi: int = 150) -> bytes:
    with pymupdf.open(pdf_path) as pdf:
        page = pdf[page_number - 1]
        return page.get_pixmap(dpi=dpi).tobytes("png")


def transcribe_page_gemini(pdf_path: Path, page_number: int) -> str:
    png_bytes = render_page_png(pdf_path, page_number)
    cache_key = json.dumps({"model": GEMINI_MODEL, "doc": pdf_path.name, "page": page_number, "task": "transcribe"})

    def _call() -> str:
        resp = call_with_retry_gemini(
            lambda: _gemini_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=[_TRANSCRIBE_PROMPT, types.Part.from_bytes(data=png_bytes, mime_type="image/png")],
            )
        )
        return resp.text or ""

    return cached_call(f"gemini_transcribe/{GEMINI_MODEL}", cache_key, _call)


def summarize_table_gemini(markdown: str, page_no: int) -> str:
    cache_key = json.dumps({"model": GEMINI_MODEL, "page": page_no, "markdown": markdown})

    def _call() -> str:
        resp = call_with_retry_gemini(
            lambda: _gemini_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=[_SUMMARY_PROMPT.format(markdown=markdown)],
            )
        )
        return (resp.text or "").strip()

    return cached_call(f"gemini_summarize/{GEMINI_MODEL}", cache_key, _call)


def cross_check_agreement(docling_markdown: str, gemini_markdown: str) -> float:
    docling_numbers = extract_numbers(docling_markdown)
    if not docling_numbers:
        return 1.0
    gemini_numbers = extract_numbers(gemini_markdown)
    return len(docling_numbers & gemini_numbers) / len(docling_numbers)


def build_table_records(pdf_path: Path) -> tuple[list[TableRecord], list[tuple[int, str]]]:
    """Returns (records, failures). A failure on one table doesn't lose the rest of the batch."""
    docling_tables = extract_docling_tables(pdf_path)
    records = []
    failures: list[tuple[int, str]] = []
    for dt in docling_tables:
        try:
            gemini_md = transcribe_page_gemini(pdf_path, dt.page_no)
            agreement = cross_check_agreement(dt.markdown, gemini_md)
            summary = summarize_table_gemini(dt.markdown, dt.page_no)
        except Exception as e:
            failures.append((dt.page_no, str(e)[:200]))
            continue
        records.append(
            TableRecord(
                page_no=dt.page_no,
                docling_markdown=dt.markdown,
                gemini_markdown=gemini_md,
                agreement=agreement,
                summary=summary,
            )
        )
    return records, failures

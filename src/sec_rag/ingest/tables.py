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
from google import genai
from google.genai import types

from sec_rag.cache import cached_call
from sec_rag.config import get_settings
from sec_rag.ingest.docling_extraction import extract_docling
from sec_rag.numbers import extract_numbers
from sec_rag.openai_utils import call_with_retry_gemini

GEMINI_MODEL = "gemini-3.1-flash-lite"

_TRANSCRIBE_PROMPT = """You're looking at one page from an SEC 10-Q filing, rendered as an image. Transcribe every table on this page into clean, well-formed GitHub-flavored markdown -- one table per table on the page, in the order they appear. If there's more than one, put a short heading line above each so it's clear where one ends and the next begins.

Keep the numbers exactly as printed: dollar signs, parenthesized negatives, decimal places, and footnote markers like (1) or (2) should stay attached to the cell they belong to rather than being dropped. These filings often use multi-row or merged headers (for example "Three Months Ended" spanning two date columns) -- flatten those into a single header row per column that keeps both pieces of information, instead of losing the top row.

Only transcribe what's actually visible on this page. If a table clearly continues from the previous page or onto the next, just transcribe the rows that are here -- don't guess at rows you can't see.

If this page has no table at all, respond with exactly: NO_TABLE
Otherwise, output only the markdown table(s) -- no preamble, no commentary, nothing after them either."""

_SUMMARY_PROMPT = """Write one or two plain sentences summarizing this table, to use as a search index entry: name the financial statement or note it's from, the specific line items it covers, and the exact periods or dates it covers.

Stick to what's actually in the table. Don't add interpretation, don't editorialize on whether a number is good or bad, and don't restate any figure in a different form than the table uses (no converting units or re-rounding).

Table (markdown):
{markdown}
"""


@dataclass(frozen=True)
class TableRecord:
    page_no: int
    docling_markdown: str
    gemini_markdown: str
    agreement: float
    summary: str


_gemini_client_instance: genai.Client | None = None


def _gemini_client() -> genai.Client:
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
    # hashing the prompt text itself (not a manually-bumped version number) means
    # editing the prompt always invalidates stale cached results automatically
    prompt_hash = hashlib.sha256(_TRANSCRIBE_PROMPT.encode()).hexdigest()[:12]
    cache_key = json.dumps({"model": GEMINI_MODEL, "doc": pdf_path.name, "page": page_number, "prompt": prompt_hash})

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
    prompt_hash = hashlib.sha256(_SUMMARY_PROMPT.encode()).hexdigest()[:12]
    cache_key = json.dumps({"model": GEMINI_MODEL, "page": page_no, "markdown": markdown, "prompt": prompt_hash})

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
    """Returns (records, failures). One table failure doesn't lose the rest."""
    docling_tables = extract_docling(pdf_path).tables
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

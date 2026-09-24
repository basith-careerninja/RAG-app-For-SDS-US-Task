"""Extracts per-page text from a PDF with PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Page:
    page_number: int  # 1-indexed
    text: str
    has_text_layer: bool


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    path: Path
    pages: list[Page]

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[page {p.page_number}]\n{p.text}" for p in self.pages)


def parse_pdf(path: Path, doc_id: str | None = None) -> ParsedDocument:
    path = Path(path)
    doc_id = doc_id or path.stem
    pages: list[Page] = []
    with pymupdf.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            pages.append(Page(page_number=i, text=text, has_text_layer=bool(text.strip())))
    return ParsedDocument(doc_id=doc_id, path=path, pages=pages)

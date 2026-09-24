"""Chunks a filing by its own section headings instead of blind token windows.

Heading detection is regex-based, tuned to standard 10-Q structure (PART/
Item/Note, all-caps statement titles). Font-size/bold detection was tried
first and dropped -- this filing's real headings and its table column
headers use nearly the same size/weight, so text-only signals win here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from sec_rag.ingest.parser import ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")

_HEADING_PATTERNS = [
    re.compile(r"^PART\s+[IVX]+\b", re.IGNORECASE),
    re.compile(r"^Item\s+\d+[A-Za-z]?[.\s]", re.IGNORECASE),
    re.compile(r"^Note\s+\d+\s*[–—-]"),
]

_MAX_HEADING_LEN = 150  

def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if len(line) <= _MAX_HEADING_LEN and any(p.match(line) for p in _HEADING_PATTERNS):
        return True

    core = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
    letters = [c for c in core if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and 2 <= len(core.split()) <= 12 and len(core) <= 90:
        return True
    return False


@dataclass(frozen=True)
class Section:
    heading: str
    page_start: int
    page_end: int
    text: str


_MIN_SECTION_CHARS = 50  # shorter than this is usually a table-of-contents line that
# happened to match a heading pattern -- fold it into the next real section


def _merge_trivial_sections(sections: list[Section]) -> list[Section]:
    merged: list[Section] = []
    pending_parts: list[str] = []
    pending_page_start: int | None = None

    for section in sections:
        if len(section.text) < _MIN_SECTION_CHARS:
            if pending_page_start is None:
                pending_page_start = section.page_start
            pending_parts.append(f"{section.heading}\n{section.text}".strip())
            continue
        page_start = pending_page_start if pending_page_start is not None else section.page_start
        text = "\n".join(pending_parts + [section.text]) if pending_parts else section.text
        merged.append(Section(heading=section.heading, page_start=page_start, page_end=section.page_end, text=text))
        pending_parts = []
        pending_page_start = None

    if pending_parts:
        if merged:
            last = merged[-1]
            merged[-1] = Section(
                heading=last.heading,
                page_start=last.page_start,
                page_end=last.page_end,
                text=last.text + "\n" + "\n".join(pending_parts),
            )
        else:
            merged.append(Section(heading="(document start)", page_start=pending_page_start or 1, page_end=pending_page_start or 1, text="\n".join(pending_parts)))

    return merged


def split_into_sections(doc: ParsedDocument) -> list[Section]:
    sections: list[Section] = []
    current_heading = "(document start)"
    current_lines: list[str] = []
    current_page_start = doc.pages[0].page_number if doc.pages else 1
    last_content_page = current_page_start

    def _flush(end_page: int) -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(Section(heading=current_heading, page_start=current_page_start, page_end=end_page, text=text))

    for page in doc.pages:
        if not page.text.strip():
            continue
        for line in page.text.split("\n"):
            if is_heading(line):
                _flush(last_content_page)
                current_heading = line.strip()
                current_lines = []
                current_page_start = page.page_number
            else:
                current_lines.append(line)
                last_content_page = page.page_number

    _flush(last_content_page)
    return _merge_trivial_sections(sections)


def _pack_lines_by_token_budget(text: str, budget: int) -> list[str]:
    pieces: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in text.split("\n"):
        line_tokens = len(_ENCODING.encode(line + "\n"))
        if current_lines and current_tokens + line_tokens > budget:
            pieces.append("\n".join(current_lines))
            current_lines = []
            current_tokens = 0
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        pieces.append("\n".join(current_lines))
    return pieces


def chunk_document_by_section(doc: ParsedDocument, max_tokens: int = 512) -> list["SectionChunk"]:
    sections = split_into_sections(doc)
    chunks: list[SectionChunk] = []
    idx = 0

    for section in sections:
        body_tokens = _ENCODING.encode(section.text)
        if len(body_tokens) <= max_tokens:
            chunks.append(
                SectionChunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}::section{idx}",
                    heading=section.heading,
                    text=f"{section.heading}\n{section.text}",
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
            idx += 1
            continue

        # split by whole lines, never mid-sentence, repeating the heading on each piece
        heading_tokens = _ENCODING.encode(section.heading + "\n")
        budget = max(1, max_tokens - len(heading_tokens))
        for piece_text in _pack_lines_by_token_budget(section.text, budget):
            chunks.append(
                SectionChunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}::section{idx}",
                    heading=section.heading,
                    text=f"{section.heading}\n{piece_text}",
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
            idx += 1

    return chunks


@dataclass(frozen=True)
class SectionChunk:
    doc_id: str
    chunk_id: str
    heading: str
    text: str
    page_start: int
    page_end: int

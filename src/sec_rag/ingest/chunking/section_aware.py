"""Chunks a filing using Docling's own section-header classification
(from its layout model) rather than guessing headings from plain text.

An earlier regex-based version matched "Item N." patterns on raw text and
had a real bug: a plain sentence that happened to start with "Item 1A of
the Company's..." got misclassified as a heading, silently merging and
mislabeling a large chunk of the filing. Docling's layout model looks at
actual visual structure (position, font, etc.) to classify each text block,
so it doesn't make that kind of mistake, and it picks up finer-grained
subsections (individual product lines, segment names) that the regex
approach couldn't reliably catch either.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from sec_rag.ingest.docling_extraction import DoclingTextItem

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SKIP_LABELS = {"page_header", "page_footer"}  # repeated boilerplate, not content

# Signature/certification pages repeat the company name and filing date
# verbatim (e.g. "...Quarterly Report of Apple Inc. ... for the period
# ended June 25, 2022...") with zero actual financial content -- that
# overlap was enough to make a CEO certification outrank the real balance
# sheet table for "what was Apple's total assets at June 25, 2022". They
# never answer a real question, so they're dropped from the index entirely
# rather than left in to compete on keyword overlap.
_BOILERPLATE_HEADING_MARKERS = ("signature", "certification", "sarbanes-oxley", "18 u.s.c.")


def _is_boilerplate(heading: str) -> bool:
    heading = heading.lower()
    return any(marker in heading for marker in _BOILERPLATE_HEADING_MARKERS)


@dataclass(frozen=True)
class Section:
    heading: str
    page_start: int
    page_end: int
    text: str


def split_into_sections(text_items: list[DoclingTextItem]) -> list[Section]:
    sections: list[Section] = []
    current_heading = "(document start)"
    current_lines: list[str] = []
    current_page_start = text_items[0].page_no if text_items else 1
    last_content_page = current_page_start

    def _flush(end_page: int) -> None:
        text = "\n".join(current_lines).strip()
        if text and not _is_boilerplate(current_heading):
            sections.append(Section(heading=current_heading, page_start=current_page_start, page_end=end_page, text=text))

    for item in text_items:
        if item.label in _SKIP_LABELS:
            continue
        if item.label == "section_header":
            _flush(last_content_page)
            current_heading = item.text.strip()
            current_lines = []
            current_page_start = item.page_no
        else:
            current_lines.append(item.text)
            last_content_page = item.page_no

    _flush(last_content_page)
    return sections


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


def chunk_document_by_section(doc_id: str, text_items: list[DoclingTextItem], max_tokens: int = 512) -> list["SectionChunk"]:
    sections = split_into_sections(text_items)
    chunks: list[SectionChunk] = []
    idx = 0

    for section in sections:
        body_tokens = _ENCODING.encode(section.text)
        if len(body_tokens) <= max_tokens:
            chunks.append(
                SectionChunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::section{idx}",
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
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::section{idx}",
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

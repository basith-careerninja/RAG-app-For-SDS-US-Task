"""Fixed-size token chunker with overlap -- the naive baseline chunking strategy."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from sec_rag.ingest.parser import ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int


def chunk_document(doc: ParsedDocument, size: int = 512, overlap: int = 50) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    flat_tokens: list[int] = []
    flat_pages: list[int] = []
    for page in doc.pages:
        if not page.text.strip():
            continue
        tokens = _ENCODING.encode(page.text)
        flat_tokens.extend(tokens)
        flat_pages.extend([page.page_number] * len(tokens))

    if not flat_tokens:
        return []

    stride = size - overlap
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(flat_tokens):
        end = min(start + size, len(flat_tokens))
        window_tokens = flat_tokens[start:end]
        window_pages = flat_pages[start:end]
        text = _ENCODING.decode(window_tokens)
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_id=f"{doc.doc_id}::chunk{idx}",
                text=text,
                page_start=min(window_pages),
                page_end=max(window_pages),
            )
        )
        idx += 1
        if end == len(flat_tokens):
            break
        start += stride

    return chunks

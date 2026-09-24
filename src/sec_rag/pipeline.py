"""Builds a runnable pipeline from an experiment config dict."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from sec_rag.cache import MISSING, cache_get, cache_set
from sec_rag.doc_catalog import DocCatalog
from sec_rag.generate.generator import ContextItem, Generator
from sec_rag.index.embedders.openai_embedder import OpenAIEmbedder
from sec_rag.index.stores.inmemory import InMemoryVectorStore
from sec_rag.ingest.chunking.fixed import chunk_document
from sec_rag.ingest.chunking.section_aware import chunk_document_by_section
from sec_rag.ingest.parser import parse_pdf
from sec_rag.retrieve.dense import DenseRetriever
from sec_rag.retrieve.hybrid import HybridRetriever

DEFAULT_GENERATOR_MODEL = "gpt-5.4-nano"


class Pipeline(Protocol):
    def answer(self, question: str, doc_ids: list[str]):
        ...


class NaiveRagPipeline:
    """Parse -> chunk -> embed -> retrieve -> generate. Chunker and retriever
    are picked by config, so the naive floor baseline (fixed-token chunks,
    dense retrieval) and the section-aware + hybrid version share this class.

    With section_aware chunking, each table gets a short LLM summary
    embedded for search and its full markdown swapped in as context once
    that summary is the retrieved hit (multi-vector / parent-document
    pattern).
    """

    def __init__(self, config: dict, docs_dir: Path):
        self.config = config
        self.catalog = DocCatalog(docs_dir)
        chunk_cfg = config.get("chunker", {})
        self.chunker_type = chunk_cfg.get("type", "fixed_tokens")
        self.chunk_size = chunk_cfg.get("size", 512)
        self.chunk_overlap = chunk_cfg.get("overlap", 50)
        self.retriever_type = config.get("retriever", {}).get("type", "dense")
        self.top_k = config.get("retriever", {}).get("top_k", 5)

        embed_cfg = config.get("embedder", {})
        self.embedder = OpenAIEmbedder(model=embed_cfg.get("model", "text-embedding-3-small"))
        self.generator = Generator(model=config.get("generator", {}).get("model", DEFAULT_GENERATOR_MODEL))

        self.store = InMemoryVectorStore()
        search_texts = self._ingest_corpus()

        if self.retriever_type == "hybrid":
            self.retriever = HybridRetriever(self.store, self.embedder, search_texts)
        else:
            self.retriever = DenseRetriever(self.store, self.embedder)

    def _index_cache_key(self) -> str:
        # Hashing the actual source of the modules that produce the index (not a
        # manually-bumped version number) means editing chunking/table logic or
        # prompts always invalidates a stale cached index automatically, instead
        # of silently serving results built with old code.
        file_hashes = sorted(hashlib.sha256(e.path.read_bytes()).hexdigest() for e in self.catalog.entries)
        parts = [
            *file_hashes,
            self.chunker_type,
            str(self.chunk_size),
            str(self.chunk_overlap),
            self.embedder.model,
        ]
        if self.chunker_type == "section_aware":
            import sec_rag.ingest.chunking.section_aware as section_aware_module
            import sec_rag.ingest.docling_extraction as docling_extraction_module
            import sec_rag.ingest.tables as tables_module

            for module in (section_aware_module, docling_extraction_module, tables_module):
                parts.append(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        else:
            import sec_rag.ingest.chunking.fixed as fixed_module

            parts.append(hashlib.sha256(Path(fixed_module.__file__).read_bytes()).hexdigest())
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def _ingest_corpus(self) -> list[str]:
        cache_key = self._index_cache_key()
        cached = cache_get("pipeline_index", cache_key)
        if cached is not MISSING:
            for chunk_id, vector, metadata in cached["entries"]:
                self.store.add(chunk_id, vector, metadata)
            return cached["search_texts"]

        items = self._build_items()
        search_texts = [item[1] for item in items]
        vectors = self.embedder.embed_many(search_texts)

        entries = []
        for (chunk_id, _search_text, context_text, doc_id, page_start, page_end), vector in zip(items, vectors):
            metadata = {"text": context_text, "doc_id": doc_id, "page_start": page_start, "page_end": page_end}
            self.store.add(chunk_id, vector, metadata)
            entries.append((chunk_id, vector, metadata))

        cache_set("pipeline_index", cache_key, {"entries": entries, "search_texts": search_texts})
        return search_texts

    def _build_items(self) -> list[tuple[str, str, str, str, int, int]]:
        # search_text is what gets embedded/indexed; context_text is what's
        # handed to the generator -- they only differ for table chunks
        items: list[tuple[str, str, str, str, int, int]] = []

        for entry in self.catalog.entries:
            if self.chunker_type == "section_aware":
                # lazy imports: pull in docling/torch only when the index cache misses
                from sec_rag.ingest.docling_extraction import extract_docling
                from sec_rag.ingest.tables import build_table_records

                text_items = extract_docling(entry.path).text_items
                for chunk in chunk_document_by_section(entry.doc_id, text_items, max_tokens=self.chunk_size):
                    items.append((chunk.chunk_id, chunk.text, chunk.text, chunk.doc_id, chunk.page_start, chunk.page_end))

                table_records, _failures = build_table_records(entry.path)
                for i, rec in enumerate(table_records):
                    chunk_id = f"{entry.doc_id}::table_p{rec.page_no}_{i}"
                    items.append((chunk_id, rec.summary, rec.docling_markdown, entry.doc_id, rec.page_no, rec.page_no))
            else:
                doc = parse_pdf(entry.path, doc_id=entry.doc_id)
                for chunk in chunk_document(doc, size=self.chunk_size, overlap=self.chunk_overlap):
                    items.append((chunk.chunk_id, chunk.text, chunk.text, chunk.doc_id, chunk.page_start, chunk.page_end))

        return items

    def answer(self, question: str, doc_ids: list[str]):
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        context_items = [ContextItem(label=f"{c.doc_id}.pdf, p.{c.page_start}-{c.page_end}", text=c.text) for c in retrieved]
        answer = self.generator.answer(question, context_items)
        return answer, retrieved


class LongContextOraclePipeline:
    """No retrieval -- stuffs the full text of the requested filing(s) into the prompt."""

    def __init__(self, config: dict, docs_dir: Path):
        self.config = config
        self.catalog = DocCatalog(docs_dir)
        self.generator = Generator(model=config.get("generator", {}).get("model", DEFAULT_GENERATOR_MODEL))

    def answer(self, question: str, doc_ids: list[str]):
        context_items = []
        for doc_id in doc_ids:
            doc = parse_pdf(self.catalog.path_for(doc_id), doc_id=doc_id)
            context_items.append(ContextItem(label=f"{doc_id}.pdf", text=doc.full_text))
        answer = self.generator.answer(question, context_items)
        return answer, []


def build_pipeline(config: dict, docs_dir: Path) -> Pipeline:
    kind = config["pipeline"]
    if kind == "naive_rag":
        return NaiveRagPipeline(config, docs_dir)
    if kind == "long_context_oracle":
        return LongContextOraclePipeline(config, docs_dir)
    raise ValueError(f"Unknown pipeline kind: {kind!r}")

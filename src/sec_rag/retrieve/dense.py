"""Plain dense (cosine-similarity) retriever."""

from __future__ import annotations

from dataclasses import dataclass

from sec_rag.index.embedders.openai_embedder import OpenAIEmbedder
from sec_rag.index.stores.inmemory import InMemoryVectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    page_start: int
    page_end: int


class DenseRetriever:
    def __init__(self, store: InMemoryVectorStore, embedder: OpenAIEmbedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed(query)
        hits = self.store.search(query_vector, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=meta["text"],
                score=score,
                doc_id=meta["doc_id"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
            )
            for chunk_id, score, meta in hits
        ]

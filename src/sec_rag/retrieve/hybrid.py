"""Hybrid dense + BM25 retrieval, fused by Reciprocal Rank Fusion.

Ranks the whole corpus by both dense cosine similarity and BM25, then
fuses by rank rather than raw score, since the two scores aren't on
comparable scales.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from sec_rag.index.embedders.openai_embedder import OpenAIEmbedder
from sec_rag.index.stores.inmemory import InMemoryVectorStore

RRF_K = 60  # standard RRF constant


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str  # what gets passed to the generator -- the full parent content, not necessarily what matched
    score: float
    doc_id: str
    page_start: int
    page_end: int


class HybridRetriever:
    def __init__(self, store: InMemoryVectorStore, embedder: OpenAIEmbedder, search_texts: list[str]):
        # search_texts must line up with store.ids: what each entry was embedded/indexed on
        if len(search_texts) != len(store.ids):
            raise ValueError("search_texts must be parallel to store.ids")
        self.store = store
        self.embedder = embedder
        self.search_texts = search_texts
        self._bm25 = BM25Okapi([t.lower().split() for t in search_texts]) if search_texts else None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        n = len(self.store.ids)
        if n == 0:
            return []

        query_vector = self.embedder.embed(query)
        dense_hits = self.store.search(query_vector, top_k=n)  # full ranking
        dense_rank = {chunk_id: rank for rank, (chunk_id, _score, _meta) in enumerate(dense_hits, start=1)}

        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranked_ids = [self.store.ids[i] for i in sorted(range(n), key=lambda i: -bm25_scores[i])]
        bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ranked_ids, start=1)}

        rrf_scores = {
            chunk_id: 1.0 / (RRF_K + dense_rank[chunk_id]) + 1.0 / (RRF_K + bm25_rank[chunk_id])
            for chunk_id in self.store.ids
        }

        ranked_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:top_k]
        meta_by_id = {chunk_id: meta for chunk_id, _v, meta in zip(self.store.ids, self.store.vectors, self.store.metadata)}

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=meta_by_id[chunk_id]["text"],
                score=rrf_scores[chunk_id],
                doc_id=meta_by_id[chunk_id]["doc_id"],
                page_start=meta_by_id[chunk_id]["page_start"],
                page_end=meta_by_id[chunk_id]["page_end"],
            )
            for chunk_id in ranked_ids
        ]

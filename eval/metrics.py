"""Retrieval and ops metrics that don't need an LLM call (the correctness
judge, which does, lives in eval/judges/correctness.py)."""

from __future__ import annotations

from dataclasses import dataclass

from sec_rag.numbers import extract_numbers


def faithfulness_score(answer_text: str, context_text: str) -> float:
    """Fraction of numbers in the answer that also appear in the context.
    1.0 if the answer makes no numeric claims."""
    answer_numbers = extract_numbers(answer_text)
    if not answer_numbers:
        return 1.0
    context_numbers = extract_numbers(context_text)
    supported = sum(1 for n in answer_numbers if n in context_numbers)
    return supported / len(answer_numbers)


def doc_recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 1.0
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for d in gold_doc_ids if d in retrieved_set)
    return hits / len(gold_doc_ids)


def mrr(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    gold_set = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def silver_page_recall(retrieved_pages: list[tuple[str, int]], silver_pages: list[tuple[str, int]]) -> float:
    if not silver_pages:
        return 1.0
    retrieved_set = set(retrieved_pages)
    hits = sum(1 for p in silver_pages if p in retrieved_set)
    return hits / len(silver_pages)


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    mean_ms: float


def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, mean_ms=0.0)
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def _pct(p: float) -> float:
        idx = min(n - 1, int(round(p * (n - 1))))
        return sorted_lat[idx]

    return LatencyStats(p50_ms=_pct(0.50), p95_ms=_pct(0.95), mean_ms=sum(sorted_lat) / n)

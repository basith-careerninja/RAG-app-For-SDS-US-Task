"""Figures out which page(s) of the filing each eval question's answer
actually comes from, for a page-level retrieval recall metric.

Two methods:
- "heuristic" (default, free): matches numbers in the gold answer against
  numbers on each page. Works well for table-sourced questions, weaker for
  qualitative text answers. Tags each result high/low/unresolved confidence.
- "llm" (opt-in, costs a little): asks a model to read the filing and pick
  the page(s), for whatever the heuristic couldn't confidently resolve.

Usage:
    python -m eval.silver_pages --split aapl_dev
    python -m eval.silver_pages --split aapl_dev --method llm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

from sec_rag.cache import cached_call
from sec_rag.config import get_settings
from sec_rag.doc_catalog import DocCatalog
from sec_rag.ingest.parser import parse_pdf
from sec_rag.numbers import extract_numbers
from sec_rag.openai_utils import RequestTooLargeError, call_with_retry

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "data" / "raw" / "aapl"
DEFAULT_LLM_MODEL = "gpt-5.4-nano"

PROMPT_TEMPLATE = """You are given the full text of one or more SEC 10-Q filings, each page marked [page N], and a question with its reference answer. List which page number(s) in EACH filing actually contain the information used to answer the question.

Question: {question}
Reference answer: {gold_answer}

{filings_block}

Respond with strict JSON only: {{"<doc_id>": [<page numbers>], ...}} using exactly the doc_id labels given above. Only include pages that materially support the answer, not every page mentioning the topic in passing.
"""


def label_question_heuristic(question: str, gold_answer: str, doc_ids: list[str], catalog: DocCatalog) -> dict:
    gold_numbers = extract_numbers(gold_answer)
    if not gold_numbers:
        return {"pages": {}, "confidence": "unresolved", "reason": "no numeric content in gold answer", "gold_numbers": []}

    page_scores: list[tuple[str, int, int]] = []
    for doc_id in doc_ids:
        doc = parse_pdf(catalog.path_for(doc_id), doc_id=doc_id)
        for page in doc.pages:
            match_count = len(gold_numbers & extract_numbers(page.text))
            if match_count > 0:
                page_scores.append((doc_id, page.page_number, match_count))

    if not page_scores:
        return {
            "pages": {},
            "confidence": "unresolved",
            "reason": "none of the gold answer's numbers appear on any candidate page",
            "gold_numbers": sorted(gold_numbers),
        }

    max_count = max(c for _, _, c in page_scores)
    top_pages = [(d, p) for d, p, c in page_scores if c == max_count]

    pages_by_doc: dict[str, list[int]] = {}
    for d, p in top_pages:
        pages_by_doc.setdefault(d, []).append(p)
    for d in pages_by_doc:
        pages_by_doc[d].sort()

    coverage = max_count / len(gold_numbers)
    confidence = "high" if len(top_pages) <= 3 and coverage >= 0.5 else "low"

    return {
        "pages": pages_by_doc,
        "confidence": confidence,
        "reason": f"{max_count}/{len(gold_numbers)} gold numbers matched, on {len(top_pages)} page(s)",
        "gold_numbers": sorted(gold_numbers),
    }


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def normalize_pages_by_doc(pages_by_doc: dict, valid_doc_ids: list[str]) -> dict[str, list[int]]:
    # the model doesn't always echo doc_id labels exactly (sometimes appends ".pdf")
    valid_set = set(valid_doc_ids)
    normalized: dict[str, list[int]] = {}
    for key, pages in pages_by_doc.items():
        candidate = key[:-4] if key.lower().endswith(".pdf") else key
        if candidate not in valid_set:
            continue
        normalized.setdefault(candidate, [])
        normalized[candidate] = sorted(set(normalized[candidate]) | set(pages))
    return normalized


def label_question_llm(question: str, gold_answer: str, doc_ids: list[str], catalog: DocCatalog, model: str) -> dict[str, list[int]]:
    filings_block = ""
    for doc_id in doc_ids:
        doc = parse_pdf(catalog.path_for(doc_id), doc_id=doc_id)
        filings_block += f"=== {doc_id} ===\n{doc.full_text}\n\n"

    prompt = PROMPT_TEMPLATE.format(question=question, gold_answer=gold_answer, filings_block=filings_block)
    cache_key = json.dumps({"model": model, "question": question, "doc_ids": doc_ids}, sort_keys=True)

    def _call() -> dict:
        resp = call_with_retry(
            lambda: _client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    raw = cached_call(f"silver_pages/{model}", cache_key, _call)
    return normalize_pages_by_doc(raw, catalog.doc_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="aapl_dev", help="basename of data/eval/<split>.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="only used with --method llm")
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()

    if "test" in args.split.lower() and not args.unlock_test:
        raise SystemExit(f"Refusing to run on a split named {args.split!r} without --unlock-test.")

    split_csv = REPO_ROOT / "data" / "eval" / f"{args.split}.csv"
    df = pd.read_csv(split_csv)
    if args.limit:
        df = df.head(args.limit)

    catalog = DocCatalog(DOCS_DIR)
    out_path = REPO_ROOT / "data" / "eval" / f"{args.split}_silver_pages.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # a record counts as trusted (skip re-labeling) if it's high-confidence
    # heuristic or already LLM-labeled; low-confidence ones can be replaced
    # by a later --method llm pass
    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                existing[record["question"]] = record

    def is_trusted(record: dict) -> bool:
        return record.get("confidence") == "high" or record.get("method") == "llm"

    needs_review: list[dict] = []
    skipped_too_large: list[str] = []

    for _, row in df.iterrows():
        question = row["Question"]
        if question in existing and is_trusted(existing[question]):
            continue
        doc_ids = row["doc_ids"].split(";")

        if args.method == "heuristic":
            result = label_question_heuristic(question, row["Answer"], doc_ids, catalog)
            record = {"question": question, "doc_ids": doc_ids, "method": "heuristic", **result}
            if result["confidence"] != "high":
                needs_review.append({"question": question, **result})
        else:
            try:
                pages_by_doc = label_question_llm(question, row["Answer"], doc_ids, catalog, args.model)
            except RequestTooLargeError:
                skipped_too_large.append(question)
                print(f"SKIPPED (request too large): {question[:70]!r}")
                continue
            record = {"question": question, "doc_ids": doc_ids, "method": "llm", "model": args.model, "pages": pages_by_doc}

        existing[question] = record
        print(f"labeled ({record.get('confidence', 'llm')}): {question[:70]!r} -> {record['pages']}")

    with open(out_path, "w", encoding="utf-8") as out:
        for record in existing.values():
            out.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(existing)} total records to {out_path}")

    if needs_review:
        print(f"\n{len(needs_review)} question(s) need manual review:")
        for r in needs_review:
            print(f"  - [{r['confidence']}] {r['question'][:90]!r} -- {r['reason']}")

    if skipped_too_large:
        print(f"\n{len(skipped_too_large)} question(s) skipped (request too large):")
        for q in skipped_too_large:
            print(f"  - {q[:100]}")


if __name__ == "__main__":
    main()

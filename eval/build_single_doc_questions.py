"""Drafts extra Q&A pairs for the filing using an LLM, since the upstream
eval set only has 2 questions answerable from this one document. Output is
a draft -- meant to be reviewed and edited by hand before use.

Usage:
    python -m eval.build_single_doc_questions --num-questions 22
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openai import OpenAI

from sec_rag.cache import cached_call
from sec_rag.config import get_settings
from sec_rag.ingest.parser import parse_pdf
from sec_rag.openai_utils import call_with_retry

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_DOC = REPO_ROOT / "data" / "raw" / "aapl" / "2022 Q3 AAPL.pdf"

PROMPT_TEMPLATE = """You are drafting a benchmark question set for a document-retrieval system, from a single SEC 10-Q filing (only this one document is in scope -- do not invent comparisons to other filings or periods).

Draft exactly {num_questions} questions with reference answers, covering a MIX of:
- questions answerable from a single passage of the filing's text
- questions answerable from a single table/structured data in the filing
- a few questions that require synthesizing 2-3 different passages within this same filing

For each question, output an object with:
- "question": the question text
- "question_type": "Single-Doc Single-Chunk" or "Single-Doc Multi-Chunk"
- "source_chunk_type": "Text" or "Table"
- "answer": a complete reference answer, quoting the specific figures/facts used

Filing (doc_id: {doc_id}):
{doc_text}

Respond with strict JSON only: {{"questions": [ ... ]}}
"""


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def draft_questions(doc_path: Path, num_questions: int, model: str) -> list[dict]:
    doc = parse_pdf(doc_path)
    prompt = PROMPT_TEMPLATE.format(num_questions=num_questions, doc_id=doc.doc_id, doc_text=doc.full_text)
    cache_key = json.dumps({"model": model, "doc_id": doc.doc_id, "num_questions": num_questions}, sort_keys=True)

    def _call() -> dict:
        resp = call_with_retry(
            lambda: _client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            ),
            max_retries=5,
            base_delay_s=15.0,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    result = cached_call(f"single_doc_questions/{model}", cache_key, _call)
    return result.get("questions", [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=str(DEFAULT_DOC))
    parser.add_argument("--num-questions", type=int, default=22)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    doc_path = Path(args.doc)
    doc_id = doc_path.stem
    questions = draft_questions(doc_path, args.num_questions, args.model)

    out_path = REPO_ROOT / "data" / "eval" / f"{doc_id}_supplemental_draft.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Question", "Source Docs", "Question Type", "Source Chunk Type", "Answer"]
        )
        writer.writeheader()
        for q in questions:
            writer.writerow(
                {
                    "Question": q.get("question", ""),
                    "Source Docs": f"*{doc_id}*",
                    "Question Type": q.get("question_type", ""),
                    "Source Chunk Type": q.get("source_chunk_type", ""),
                    "Answer": q.get("answer", ""),
                }
            )

    print(f"Drafted {len(questions)} questions -> {out_path}")
    print("Not verified yet -- review every row by hand before treating this as ground truth.")


if __name__ == "__main__":
    main()

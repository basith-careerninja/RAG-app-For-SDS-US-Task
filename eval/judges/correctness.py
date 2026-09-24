"""LLM judge that scores a candidate answer against a reference answer on a 0 / 0.5 / 1 scale."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from sec_rag.cache import cached_call
from sec_rag.config import get_settings

JUDGE_MODEL = "gpt-5.4-nano"
RUBRIC_VERSION = "v1"

RUBRIC_PROMPT = """You are grading a candidate answer to a question about SEC filings, against a reference answer.

Score the candidate:
- 1.0 (correct): every number and period the reference requires is present and matches within rounding; units and signs are correct; all periods the question asks about are covered. Extra correct detail beyond the reference must NOT be penalized.
- 0.5 (partially correct): some but not all required facts/periods are right, OR there is a minor numeric/rounding/unit slip, OR a period is missing.
- 0.0 (incorrect): the candidate contradicts the reference, uses the wrong period/company, invents a number not supported by the reference, or abstains/refuses when the reference shows an answer was available.

Question: {question}

Reference answer:
{gold_answer}

Candidate answer:
{candidate_answer}

Respond with strict JSON only, no other text: {{"score": <0.0 or 0.5 or 1.0>, "reason": "<one or two sentences>"}}
"""


@dataclass
class JudgeResult:
    score: float
    reason: str


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def judge_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    model: str = JUDGE_MODEL,
) -> JudgeResult:
    cache_key = json.dumps(
        {
            "rubric_version": RUBRIC_VERSION,
            "model": model,
            "question": question,
            "gold_answer": gold_answer,
            "candidate_answer": candidate_answer,
        },
        sort_keys=True,
    )

    def _call() -> dict:
        prompt = RUBRIC_PROMPT.format(
            question=question, gold_answer=gold_answer, candidate_answer=candidate_answer
        )
        resp = _client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return {"score": float(parsed["score"]), "reason": str(parsed.get("reason", ""))}

    result = cached_call(f"judge/{model}/{RUBRIC_VERSION}", cache_key, _call)
    return JudgeResult(score=result["score"], reason=result["reason"])


def judge_agreement(judge_scores: list[float], human_scores: list[float]) -> float:
    """Fraction of exact matches between judge and human scores, for calibrating the rubric."""
    if not judge_scores:
        return 0.0
    if len(judge_scores) != len(human_scores):
        raise ValueError("judge_scores and human_scores must be the same length")
    matches = sum(1 for j, h in zip(judge_scores, human_scores) if j == h)
    return matches / len(judge_scores)

"""Builds the answer prompt and calls the model, with citations and an abstain rule."""

from __future__ import annotations

import time
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from sec_rag.config import get_settings
from sec_rag.costs import estimate_cost
from sec_rag.openai_utils import call_with_retry

DEFAULT_MODEL = "gpt-5.4-nano"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided filing excerpts. "
    "Every factual claim must be traceable to the excerpts. "
    "Cite the source filing and page number for every number or fact you use, "
    "in the form (SOURCE: <filing>, p.<page>). "
    "If the excerpts do not contain enough information to answer, say so explicitly "
    "instead of guessing."
)

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class ContextItem:
    label: str  # e.g. "2022 Q3 AAPL.pdf, p.12"
    text: str


@dataclass
class Answer:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


def build_prompt(question: str, context_items: list[ContextItem]) -> str:
    context_block = "\n\n".join(f"[{c.label}]\n{c.text}" for c in context_items)
    return f"Filing excerpts:\n\n{context_block}\n\nQuestion: {question}\n\nAnswer:"


class Generator:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=get_settings().openai_api_key)
        return self._client

    def answer(self, question: str, context_items: list[ContextItem]) -> Answer:
        prompt = build_prompt(question, context_items)
        start = time.monotonic()
        resp = call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        )
        latency_ms = (time.monotonic() - start) * 1000

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else len(_ENCODING.encode(prompt))
        output_tokens = usage.completion_tokens if usage else len(_ENCODING.encode(text))

        return Answer(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
            latency_ms=latency_ms,
        )

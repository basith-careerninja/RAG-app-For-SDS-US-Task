"""Retry helpers for OpenAI and Gemini rate limits."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from openai import RateLimitError

T = TypeVar("T")


class RequestTooLargeError(Exception):
    """A single request exceeds the account's rate limit on its own; retrying won't help."""


def call_with_retry(fn: Callable[[], T], max_retries: int = 3, base_delay_s: float = 5.0) -> T:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            if "Request too large" in str(e):
                raise RequestTooLargeError(str(e)) from e
            if attempt == max_retries:
                raise
            time.sleep(base_delay_s * (2**attempt))
    raise AssertionError("unreachable")


_GEMINI_RETRYABLE_CODES = {429, 503}  # per-minute quota, transient overload


def call_with_retry_gemini(fn: Callable[[], T], max_retries: int = 8, base_delay_s: float = 15.0) -> T:
    from google.genai.errors import APIError

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except APIError as e:
            if e.code not in _GEMINI_RETRYABLE_CODES or attempt == max_retries:
                raise
            time.sleep(base_delay_s * (attempt + 1))
    raise AssertionError("unreachable")

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

# Free-tier Flash-Lite is quoted at 15 RPM by most sources (Google doesn't
# publish a fixed number, it depends on account tier). Spacing calls out
# proactively avoids the slow 429-then-backoff cycle almost entirely --
# reacting to the limit after being throttled is much slower than just not
# exceeding it in the first place.
_GEMINI_MIN_INTERVAL_S = 5.0
_last_gemini_call_at = 0.0


def call_with_retry_gemini(fn: Callable[[], T], max_retries: int = 8, base_delay_s: float = 15.0) -> T:
    from google.genai.errors import APIError

    global _last_gemini_call_at

    for attempt in range(max_retries + 1):
        wait = _GEMINI_MIN_INTERVAL_S - (time.monotonic() - _last_gemini_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_gemini_call_at = time.monotonic()

        try:
            return fn()
        except APIError as e:
            if e.code not in _GEMINI_RETRYABLE_CODES or attempt == max_retries:
                raise
            print(f"[gemini retry] code={e.code} attempt={attempt} msg={str(e)[:150]}")
            time.sleep(base_delay_s * (attempt + 1))
    raise AssertionError("unreachable")

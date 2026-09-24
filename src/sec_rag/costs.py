"""Per-token pricing for cost tracking. Rates drift -- re-check before trusting for real spend decisions."""

from __future__ import annotations

# (input $/token, output $/token)
_PRICING_PER_TOKEN: dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.20e-6, 1.25e-6),
    "text-embedding-3-small": (0.02e-6, 0.0),
    "gemini-3.1-flash-lite": (0.25e-6, 1.50e-6),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    if model not in _PRICING_PER_TOKEN:
        raise KeyError(f"No pricing entry for {model!r}")
    in_rate, out_rate = _PRICING_PER_TOKEN[model]
    return input_tokens * in_rate + output_tokens * out_rate

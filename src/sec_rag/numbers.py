"""Pulls numeric tokens out of text, for comparing numbers across sources."""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")


def extract_numbers(text: str) -> set[str]:
    found = _NUMBER_RE.findall(text)
    normalized = set()
    for n in found:
        core = n.replace(",", "").replace("$", "").replace("%", "")
        try:
            float(core)
        except ValueError:
            continue
        normalized.add(core)
    return normalized

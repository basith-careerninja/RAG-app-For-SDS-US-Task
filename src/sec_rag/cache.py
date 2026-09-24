"""Simple disk cache keyed by content hash, used for embeddings and LLM calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_ROOT = REPO_ROOT / "data" / "cache"

MISSING = object()


def _key_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_ROOT / namespace / f"{digest}.json"


def cache_get(namespace: str, key: str) -> Any:
    path = _key_path(namespace, key)
    if not path.exists():
        return MISSING
    return json.loads(path.read_text(encoding="utf-8"))


def cache_set(namespace: str, key: str, value: Any) -> None:
    path = _key_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def cached_call(namespace: str, key: str, fn: Callable[[], Any]) -> Any:
    cached = cache_get(namespace, key)
    if cached is not MISSING:
        return cached
    result = fn()
    cache_set(namespace, key, result)
    return result

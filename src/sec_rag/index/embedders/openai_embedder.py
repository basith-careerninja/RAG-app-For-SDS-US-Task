"""OpenAI embeddings, disk-cached by (model, text)."""

from __future__ import annotations

from openai import OpenAI

from sec_rag.cache import MISSING, cache_get, cache_set, cached_call
from sec_rag.config import get_settings
from sec_rag.openai_utils import call_with_retry

DEFAULT_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


class OpenAIEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=get_settings().openai_api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        def _call() -> list[float]:
            resp = call_with_retry(lambda: self.client.embeddings.create(model=self.model, input=text))
            return resp.data[0].embedding

        return cached_call(f"embeddings/{self.model}", text, _call)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        namespace = f"embeddings/{self.model}"
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = cache_get(namespace, text)
            if cached is MISSING:
                misses.append((i, text))
            else:
                results[i] = cached

        for batch_start in range(0, len(misses), BATCH_SIZE):
            batch = misses[batch_start : batch_start + BATCH_SIZE]
            batch_texts = [t for _, t in batch]
            resp = call_with_retry(lambda: self.client.embeddings.create(model=self.model, input=batch_texts))
            for (idx, text), item in zip(batch, resp.data):
                results[idx] = item.embedding
                cache_set(namespace, text, item.embedding)

        return results

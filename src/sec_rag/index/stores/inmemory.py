"""Small in-memory vector store, cosine similarity via numpy. Fine at this corpus size."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InMemoryVectorStore:
    ids: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def add(self, id_: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self.ids.append(id_)
        self.vectors.append(vector)
        self.metadata.append(metadata)
        self._matrix = None  # invalidate cache

    def _matrix_view(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.array(self.vectors, dtype=np.float32)
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            self._matrix = self._matrix / norms
        return self._matrix

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float, dict[str, Any]]]:
        if not self.ids:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        sims = self._matrix_view() @ q
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i]), self.metadata[i]) for i in top_idx]

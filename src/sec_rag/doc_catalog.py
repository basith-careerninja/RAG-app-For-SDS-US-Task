"""Lists the PDFs in a directory and resolves a doc_id back to its path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocEntry:
    doc_id: str
    path: Path


class DocCatalog:
    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
        self.entries: list[DocEntry] = sorted(
            (DocEntry(doc_id=p.stem, path=p) for p in self.docs_dir.glob("*.pdf")),
            key=lambda e: e.doc_id,
        )
        if not self.entries:
            raise ValueError(f"No PDFs found in {self.docs_dir}")

    @property
    def doc_ids(self) -> list[str]:
        return [e.doc_id for e in self.entries]

    def path_for(self, doc_id: str) -> Path:
        for e in self.entries:
            if e.doc_id == doc_id:
                return e.path
        raise KeyError(doc_id)

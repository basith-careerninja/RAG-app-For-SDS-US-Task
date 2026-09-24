"""FastAPI app: a single filing loaded at startup, one endpoint to ask questions about it.

Run locally:
    uvicorn sec_rag.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sec_rag.config import load_experiment_config
from sec_rag.pipeline import Pipeline, build_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = REPO_ROOT / "data" / "raw" / "aapl"
PIPELINE_CONFIG_PATH = "configs/experiments/section_hybrid.yaml"
UI_DIR = REPO_ROOT / "ui"

_state: dict[str, Pipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_experiment_config(PIPELINE_CONFIG_PATH)
    _state["pipeline"] = build_pipeline(config, DOCS_DIR)
    _state["doc_id"] = next(DOCS_DIR.glob("*.pdf")).stem
    yield
    _state.clear()


app = FastAPI(title="AAPL 10-Q RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    doc_id: str
    page_start: int
    page_end: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    cost_usd: float
    latency_ms: float


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "ready": "pipeline" in _state}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse | JSONResponse:
    if not req.question.strip():
        return JSONResponse(status_code=400, content={"error": "question must not be empty"})

    pipeline = _state["pipeline"]
    answer, retrieved = pipeline.answer(req.question, [_state["doc_id"]])

    return QueryResponse(
        answer=answer.text,
        sources=[
            Source(
                doc_id=c.doc_id,
                page_start=c.page_start,
                page_end=c.page_end,
                snippet=(c.text[:300] + "...") if len(c.text) > 300 else c.text,
            )
            for c in retrieved
        ],
        model=answer.model,
        cost_usd=answer.cost_usd,
        latency_ms=answer.latency_ms,
    )


if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

# AAPL 10-Q RAG

A retrieval-augmented question-answering system over Apple's Q3 FY2022 10-Q filing. Answers questions about the filing's text and tables, with page citations.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -e .
```

Create a `.env` file (see `.env.example`):

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

Pull the filing and its reference Q&A:

```bash
python scripts/prepare_data.py --zip /path/to/KG-RAG-datasets-main.zip
```

(The zip is the [Docugami KG-RAG dataset](https://github.com/docugami/KG-RAG-datasets); only the AAPL 10-Q is extracted, into `data/raw/aapl/`.)

Run the API:

```bash
uvicorn sec_rag.api.main:app --reload
```

Open `http://127.0.0.1:8000` for the UI, or POST to `/api/query`:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was operating income for the three months ended June 25, 2022?"}'
```

## How it works

**Parsing.** Text pages go through PyMuPDF. Tables are handled separately: [Docling](https://github.com/docling-project/docling) extracts table structure, and Gemini 3.1 Flash-Lite independently transcribes the same page image to markdown as a cross-check — the two extractions are compared number-by-number, and any page where they disagree gets flagged instead of silently trusted. On this filing, all 31 tables across 21 table-bearing pages matched at 100% agreement.

**Chunking.** Text is split by the filing's own section headings (Item/Note/Part, financial statement titles) rather than blind fixed-size windows, so a paragraph never gets cut in half and separated from the sentence that answers a question. Oversized sections are split further by packing whole lines up to a token budget.

**Tables.** Each table is stored two ways: a short LLM-written summary (what gets embedded and searched against) and the full markdown table (what actually gets passed to the model once that summary is retrieved). This keeps the search index small while still giving the model the complete table when it matters.

**Retrieval.** Hybrid dense (OpenAI embeddings, cosine similarity) + BM25, combined by Reciprocal Rank Fusion.

**Generation.** A single prompt with the retrieved excerpts, instructed to cite `(SOURCE: filing, p.N)` for every claim and to say so explicitly rather than guess when the excerpts don't cover the question.

## Assumptions and limitations

- Scoped to one document by design, not a general-purpose ingestion pipeline. Ingestion logic (heading regexes, table handling) is tuned for this filing's structure and SEC 10-Q conventions generally, not arbitrary PDFs.
- This filing has no real figures or charts (its only embedded image is a small logo), so figure/chart question-answering isn't exercised here.
- No containerization yet — runs directly with `uvicorn`.

## Project layout

```
src/sec_rag/
  ingest/          parsing, chunking, table extraction
  index/           embeddings, vector store
  retrieve/        dense and hybrid retrievers
  generate/        prompting and answer generation
  api/             FastAPI app
  pipeline.py      wires the above into a runnable pipeline from a config
configs/           one YAML per pipeline variant
ui/                single-page front end
```

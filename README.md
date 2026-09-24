# RAG Task

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

**Parsing.** [Docling](https://github.com/docling-project/docling) parses the filing once and is the source of truth for both text structure and tables: its layout model classifies every text block (section header, body text, list item, page header/footer, ...) using actual visual layout, not keyword matching, and separately extracts table structure. PyMuPDF is only used for the fixed-token baseline chunker and for rendering page images for the table cross-check below.

**Tables.** Gemini 3.1 Flash-Lite independently transcribes each table page's image to markdown as a cross-check against Docling's structural extraction — the two are compared number-by-number, and any page where they disagree gets flagged instead of silently trusted. On this filing, all 31 tables across 21 table-bearing pages matched at 100% agreement.

**Chunking.** Text is split on Docling's own section-header classification (Item/Note/Part boundaries, financial statement titles, and finer subsections like individual product lines) rather than blind fixed-size windows or regex guessing, so a paragraph never gets cut in half and separated from the sentence that answers a question. An earlier regex-based version of this had a real bug — a plain sentence that happened to start with "Item 1A of the Company's..." got misread as a heading — which Docling's layout-based classification doesn't make, since it looks at how the text is actually laid out on the page, not just its wording. Oversized sections are split further by packing whole lines up to a token budget.

**Tables.** Each table is stored two ways: a short LLM-written summary (what gets embedded and searched against) and the full markdown table (what actually gets passed to the model once that summary is retrieved). This keeps the search index small while still giving the model the complete table when it matters.

**Retrieval.** Hybrid dense (OpenAI embeddings, cosine similarity) + BM25, combined by Reciprocal Rank Fusion.

**Generation.** A single prompt with the retrieved excerpts, instructed to cite `(SOURCE: filing, p.N)` for every claim and to say so explicitly rather than guess when the excerpts don't cover the question.

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

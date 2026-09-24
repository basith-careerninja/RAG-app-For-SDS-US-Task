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

(The zip is the [Docugami KG-RAG dataset](https://github.com/docugami/KG-RAG-datasets); only the AAPL 10-Q and its Q&A file are extracted, into `data/raw/aapl/`.)

Build the eval set and run the API:

```bash
python -m eval.split
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

## Evaluation

`eval/` has the harness: a stratified dev/test split, an LLM judge that scores answers 0/0.5/1 against reference answers, and retrieval metrics (recall, MRR, faithfulness — every number in an answer should trace back to the retrieved context).

Two baselines for comparison, plus the actual pipeline:

| | Naive (fixed-token chunks, dense-only) | Section-aware + hybrid |
|---|---|---|
| Correctness (18-question dev set) | 0.917 | 0.972 |
| Table questions | 0.79 | 0.93 |
| Cost per query | ~$0.0007 | ~$0.0006 |

A long-context baseline (the whole filing stuffed into the prompt, no retrieval at all) scores 1.000 on the same set — the filing is small enough that this is a legitimate, cheap alternative, not just a theoretical ceiling.

```bash
python -m eval.run_eval --config configs/experiments/section_hybrid.yaml --split aapl_dev
```

## Assumptions and limitations

- Scoped to one document by design, not a general-purpose ingestion pipeline. Ingestion logic (heading regexes, table handling) is tuned for this filing's structure and SEC 10-Q conventions generally, not arbitrary PDFs.
- This filing has no real figures or charts (its only embedded image is a small logo), so figure/chart question-answering isn't exercised here.
- The eval set is small (26 questions total: 2 from the source dataset's Q&A, the rest drafted and reviewed by hand) given how few of the original dataset's questions are answerable from this one filing alone.
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
eval/              eval-set construction, metrics, judge, report generator
configs/           one YAML per pipeline variant
ui/                single-page front end
```

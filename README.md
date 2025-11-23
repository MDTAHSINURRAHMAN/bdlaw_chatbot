# BDLaws — Legal Assistant (Semantic Search + LLM Synthesis)

A production-focused legal assistant that indexes Bangladesh legislation into a vector store, performs semantic retrieval with Jina embeddings + Qdrant, and synthesizes answers using an LLM (via OpenRouter/Gemini). Built as a FastAPI service and a set of tooling scripts for ingestion, validation, and maintenance.

## Overview

BDLaws is a toolkit and backend service for building a law-aware question-answering assistant over Bangladesh statutes. It:

- Embeds full legal sections using Jina embeddings and stores them in Qdrant for fast vector search.
- Uses semantic search results to expand context (by chapter and nearby sections).
- Invokes an LLM (via OpenRouter / Gemini models) to analyze law chunks, rate relevance, and produce a final synthesized legal summary and insights.
- Exposes a simple FastAPI endpoint (`/legal-analyze`) to accept a user query and return a structured JSON containing summaries and enriched section metadata.

This repository includes ingestion scripts (create collection, embed and upload), maintenance utilities (find missing entries, re-embed missing), data validation, and an interactive CLI flow for one-off testing.

## Key Features

- Embeddings pipeline using Jina (`jina-embeddings-v3`) to produce high-dim vectors for legal sections.
- Vector store built on Qdrant for performant nearest-neighbour search and payload storage (act metadata, URLs, full text).
- LLM-driven relevance scoring and analysis via OpenRouter / Google Gemini — used both to rank sections and to synthesize final answers.
- Context expansion by chapter and neighbor sections to provide richer legal context for LLM analysis.
- Robust ingestion tooling: collection creation (`create_qdrant_collection.py`), large-batch embedding and upload (`embed_and_upload.py`), re-embedding for missing entries (`embed_and_upload_missing.py`), and a comparator (`find_missing_laws.py`).
- FastAPI web service (`main.py`) with CORS enabled and a `Procfile` for easy deployment.

## Tech Stack & Tools

- Languages

  - Python 3.10 (runtime.txt specifies `python-3.10.14`)

- Frameworks & Libraries (from `requirements.txt`)

  - FastAPI == 0.115.13 — HTTP API
  - Pydantic == 1.10.15 — request models and validation
  - python-dotenv == 1.1.1 — load environment config
  - qdrant-client == 1.14.3 — Qdrant vector DB client
  - requests == 2.32.4 — HTTP requests to Jina / OpenRouter
  - jina == 3.34.0 — embedding model compatibility

- Infrastructure

  - Qdrant (vector database) — collection `bdlaws`
  - Jina embedding service (remote embedding endpoint) — configured via `JINA_API_URL`/`JINA_API_KEY`
  - OpenRouter (LLM proxy) / Google Gemini — configured via `OPENROUTER_API_URL`/`OPENROUTER_API_KEY`
  - Procfile + uvicorn for simple Heroku-like deployment

- Dev Tools
  - dotenv for local environment
  - tqdm for progress bars during ingest
  - tiktoken (used in `embed_and_upload_missing.py` for tokenizer-based truncation)

## Project Structure (simplified)

- README.md — (this file)
- requirements.txt — pinned Python dependencies
- runtime.txt — Python runtime (3.10.14)
- Procfile — `uvicorn main:app` for deployment

Core scripts (top-level):

- `main.py` — FastAPI app exposing `/legal-analyze` endpoint (API entrypoint).
- `core.py` — central library with: embedding call, semantic search, relevance scoring, chunk analysis, final summary synthesis, context expansion, and Qdrant helpers.
- `semantic_search.py` — near-duplicate copy of `core.py` utilities (CLI-friendly) used for local testing.
- `create_qdrant_collection.py` — creates/recreates the `bdlaws` collection with VectorParams (size=1024, cosine distance).
- `embed_and_upload.py` — full dataset embedding and upload from `bdlaws_merged.json` into Qdrant (batching, retries).
- `embed_and_upload_missing.py` — re-embed specifically missing entries (uses token-aware truncation with `tiktoken`).
- `find_missing_laws.py` — compares `bdlaws_merged.json` with Qdrant uploads and writes `missing_laws.json`.
- `merge_laws.py` — simple data validator that looks for null/NaN required fields in `bdlaws_merged.json`.

Data & outputs:

- `bdlaws_merged.json` — (expected) canonical dataset of act/section objects used for embedding (not committed in full to repo snapshot here).
- `missing_laws.json` — generated list of sections missing from Qdrant (produced by `find_missing_laws.py`).
- `final_legal_response.json` — sample LLM-produced final output for a query (included as example).
- `DATA/` — auxiliary files and index (repository contains an index and overview JSONs).

## Getting Started

Minimum prerequisites

- Python 3.10 (runtime.txt specifies 3.10.14). A modern venv is recommended.
- Access to a running Qdrant instance (cloud or local).
- A Jina embeddings endpoint (Jina Cloud or self-hosted) compatible with `jina-embeddings-v3`.
- OpenRouter / LLM API credentials (used for relevance scoring & synthesis).

Environment variables (use `.env`)

- QDRANT_URL — Qdrant endpoint (e.g. `http://localhost:6333`)
- QDRANT_API_KEY — (if applicable)
- JINA_API_URL — Jina embeddings HTTP API URL
- JINA_API_KEY — Jina API key
- OPENROUTER_API_URL — OpenRouter API base URL
- OPENROUTER_API_KEY — OpenRouter API key

Install dependencies

```powershell
# From repository root (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Create (or recreate) Qdrant collection

```powershell
# Ensure QDRANT_* env vars are set
python create_qdrant_collection.py
```

Embed & upload (full corpus)

```powershell
# Expects bdlaws_merged.json in project root
python embed_and_upload.py
```

Find and re-embed missing entries

```powershell
python find_missing_laws.py
# if missing_laws.json is non-empty
python embed_and_upload_missing.py
```

Run the API service locally

```powershell
# development run
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# or use the Procfile command on Heroku-like platforms (Procfile uses uvicorn main:app)
```

API example

POST /legal-analyze

Request body (JSON):

```json
{
  "query": "Can a person occupying ancestral land be forcibly evicted without due process?"
}
```

Successful response returns a JSON with fields like `final_summary`, `final_legal_insight`, and `relevant_sections_full` (enriched metadata from Qdrant).

## Testing

- There are no automated tests included in the repository. (No `tests/` directory or test runner configuration detected.)
- Manual checks available:
  - Run `merge_laws.py` to validate dataset fields.
  - Run `semantic_search.py` interactively for local CLI testing.
  - Use `find_missing_laws.py` and `embed_and_upload_missing.py` to verify ingestion completeness.

Suggested quick tests to add (recommended):

- Pytest unit tests for `core.py` helpers (embedding wrapper, `clean_gemini_json`, and `expand_with_related_sections`).
- Integration test that mocks Jina and OpenRouter responses and asserts `main.analyze_legal_query` returns the expected schema.

## Showcase / Screenshots

If you add a frontend or dashboard, these placeholders will be helpful:

- `screenshots/api_response_example.png` — sample JSON response shown in a formatted UI.
- `screenshots/embedding_progress.png` — progress bar during bulk upload.
- `screenshots/section_browser.png` — a UI that shows a retrieved section and LLM summary.

## Design Decisions & Architecture Notes

BDLaws is organized as a small, pragmatic pipeline combining a vector index and a language model orchestrator. The core design decisions visible in the codebase are:

- Vector-first retrieval: the system relies on Jina-generated embeddings and Qdrant as the canonical retrieval layer. This keeps the search fast and scale-friendly while storing rich payload metadata alongside each vector so the LLMs receive structured context without additional DB lookups.

- LLM-in-the-loop for both ranking and synthesis: instead of purely relying on vector similarity scores, the project sends candidate section text to an LLM (via OpenRouter/Gemini) to produce a relevance score and textual analysis. This hybrid retrieval-then-rerank approach helps surface legally-relevant passages that pure lexical/vector matching might miss.

- Chunking and context expansion: to reduce hallucination risk and to provide adequate context size to the LLM, retrieved sections are expanded by chapter and neighborhood (±N sections) and then split into chunk batches (default batch size 10) for iterative analysis. The final summary stage recombines chunk analyses into a single JSON response.

- Practical engineering choices: embedding batch sizes, retry loops for external calls, token-aware truncation using `tiktoken` (to avoid model input errors), and a 1024-dim vector size configured for Jina v3 embeddings are examples of operationally-aware decisions intended to make ingestion reliable and LLM calls durable.

## Known Omissions & Notes

- No automated test suite is included. Adding pytest-based unit and integration tests is recommended before production rollout.
- No explicit frontend is present — the project exposes an HTTP API and includes sample output (`final_legal_response.json`) for demonstration.
- Licensing: No `LICENSE` file is present in the repository, so **License not specified**. Add a license file (MIT, Apache-2.0, etc.) if you intend to open-source this project.

## Contact & Attribution

Built by MDTAHSINURRAHMAN.

If you’d like to collaborate or request changes, open an issue or contact via GitHub: https://github.com/MDTAHSINURRAHMAN

## License

License not specified in repository. Add a `LICENSE` file to declare terms.

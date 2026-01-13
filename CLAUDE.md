# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) learning project using LangChain and Pinecone to answer questions about FastAPI documentation. The project indexes FastAPI documentation markdown files into Pinecone and uses LLM-based retrieval to answer questions.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the baseline RAG pipeline
python rag_baseline.py

# Run the chunks-only RAG pipeline (with data cleaning, no parent-child strategy)
python rag_chunks_only.py

# Run the improved RAG pipeline (with data cleaning and MultiVectorRetriever)
python rag_data_cleaned.py

# Run the hybrid search + multi-query RAG pipeline
python rag_hybrid_multiquery.py

# Run RAG evaluation using RAGAS metrics
python eval.py

# Run comparative evaluation across all pipelines
python eval_hybrid.py

# Test markdown cleaning on sample data
python cleaner.py  # With code deduplication (for parent-child retrieval)
python cleaner_no_dedup.py  # Without deduplication (for chunk-based retrieval)

# Debug embedding similarity scores
python embedding_test.py

# Debug retriever results and scores
python retriever_test.py
```

## Architecture

### RAG Pipelines

Four RAG implementations exist:

1. **rag_baseline.py** - Basic RAG with RecursiveCharacterTextSplitter (chunk_size=2000, overlap=400)
2. **rag_chunks_only.py** - Improved RAG using:
   - `cleaner_no_dedup.py` for markdown preprocessing (removes badges, extracts ALL code - no deduplication)
   - MarkdownHeaderTextSplitter (splits by `##` headers)
   - Direct chunk storage in Pinecone (no parent-child strategy)
   - Pinecone index: `fastapi-chunks-only`
3. **rag_data_cleaned.py** - Advanced RAG using:
   - `cleaner.py` for markdown preprocessing (removes badges, extracts code with deduplication)
   - MarkdownHeaderTextSplitter (splits by `##` headers)
   - MultiVectorRetriever with parent-child document strategy (child chunks for retrieval, parent docs for context)
   - Local file store (`parent_docs_store/`) for parent documents
   - Pinecone index: `fastapi-cleaned`
4. **rag_hybrid_multiquery.py** - Production-grade RAG using:
   - `cleaner_no_dedup.py` for markdown preprocessing (removes badges, extracts ALL code - no deduplication)
   - MarkdownHeaderTextSplitter (splits by `##` headers)
   - **Hybrid Search**: Pinecone native hybrid search combining dense vectors + sparse BM25 vectors
   - **Multi-Query**: LLM-generated query variations (3-5 variations) to improve recall
   - BM25Encoder with jieba tokenization for Chinese text
   - Both dense and sparse vectors stored in single Pinecone index with `metric="dotproduct"`
   - Pinecone index: `fastapi-hybrid-multiquery`
   - Saved BM25 encoder: `bm25_encoder.pkl`

### Data Cleaning

Two versions of the markdown cleaner exist:

1. **cleaner.py** (with deduplication):
   - Tracks which code files have been included via `{* ... *}` tags
   - If the same code file appears multiple times in a document, only the first occurrence includes the full code
   - Subsequent occurrences show: `*[Ref: Code file is already included above]*`
   - **Use case**: Parent-child retrieval (rag_data_cleaned.py) where parent documents provide full context
   - Reduces redundancy in full documents

2. **cleaner_no_dedup.py** (without deduplication):
   - Does NOT track code file inclusion - every `{* ... *}` tag gets replaced with actual code
   - Same code file appearing in multiple chunks = code included in each chunk
   - **Use case**: Chunk-based retrieval (rag_chunks_only.py, rag_hybrid_multiquery.py) where chunks must be self-contained
   - Ensures each chunk has all necessary code examples

Both cleaners perform the same HTML cleaning:
- Remove badges, scripts, styles, SVG elements
- Extract alt text from images or remove decorative images
- Remove formatting tags while preserving content
- Extract code from `{* ... *}` file includes

### External Services

- **OpenRouter API** (`https://openrouter.ai/api/v1`): Used for both embeddings and LLM calls
- **Pinecone**: Vector database for document embeddings
- **LangSmith** (optional): Tracing and monitoring

### Key Configuration

- Pinecone index names: `PINECONE_INDEX_NAME` env var for baseline, `fastapi-chunks-only` for chunks-only, `fastapi-cleaned` for parent-child pipeline
- Embedding models: `qwen/qwen3-embedding-8b` (4096 dim) or `openai/text-embedding-3-large` (3072 dim)
- LLM: `deepseek/deepseek-r1-0528:free` via OpenRouter

### Evaluation

- **eval.py**: Evaluates a single RAG pipeline using RAGAS framework with metrics: ContextPrecision, Faithfulness, ResponseRelevancy. Results exported to CSV.
- **eval_hybrid.py**: Comparative evaluation across all pipelines (chunks_only, data_cleaned, hybrid_multiquery). Generates detailed per-query results and summary statistics comparing all approaches. Includes latency measurements.

## Environment Setup

Copy `env_template` to `.env` and fill in:
- `OPENAI_API_KEY` - OpenRouter API key
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- Optional: LangSmith keys for tracing

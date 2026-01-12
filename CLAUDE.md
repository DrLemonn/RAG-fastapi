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

# Run RAG evaluation using RAGAS metrics
python eval.py

# Test markdown cleaning on sample data
python cleaner.py

# Debug embedding similarity scores
python embedding_test.py

# Debug retriever results and scores
python retriever_test.py
```

## Architecture

### RAG Pipelines

Three RAG implementations exist:

1. **rag_baseline.py** - Basic RAG with RecursiveCharacterTextSplitter (chunk_size=2000, overlap=400)
2. **rag_chunks_only.py** - Improved RAG using:
   - `cleaner.py` for markdown preprocessing (removes badges, extracts code from `{* ... *}` includes)
   - MarkdownHeaderTextSplitter (splits by `##` headers)
   - Direct chunk storage in Pinecone (no parent-child strategy)
   - Pinecone index: `fastapi-chunks-only`
3. **rag_data_cleaned.py** - Advanced RAG using:
   - `cleaner.py` for markdown preprocessing (removes badges, extracts code from `{* ... *}` includes)
   - MarkdownHeaderTextSplitter (splits by `##` headers)
   - MultiVectorRetriever with parent-child document strategy (child chunks for retrieval, parent docs for context)
   - Local file store (`parent_docs_store/`) for parent documents
   - Pinecone index: `fastapi-cleaned`

### External Services

- **OpenRouter API** (`https://openrouter.ai/api/v1`): Used for both embeddings and LLM calls
- **Pinecone**: Vector database for document embeddings
- **LangSmith** (optional): Tracing and monitoring

### Key Configuration

- Pinecone index names: `PINECONE_INDEX_NAME` env var for baseline, `fastapi-chunks-only` for chunks-only, `fastapi-cleaned` for parent-child pipeline
- Embedding models: `qwen/qwen3-embedding-8b` (4096 dim) or `openai/text-embedding-3-large` (3072 dim)
- LLM: `deepseek/deepseek-r1-0528:free` via OpenRouter

### Evaluation

`eval.py` uses RAGAS framework with metrics: ContextPrecision, Faithfulness, ResponseRelevancy. Results exported to CSV.

## Environment Setup

Copy `env_template` to `.env` and fill in:
- `OPENAI_API_KEY` - OpenRouter API key
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- Optional: LangSmith keys for tracing

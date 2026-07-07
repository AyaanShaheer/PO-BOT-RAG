# PoBot RAG – AI Assistant for Hong Kong Labour Regulations

PoBot RAG is a production-oriented Retrieval-Augmented Generation (RAG) system that answers questions about Hong Kong labour regulations using only official government sources.

It is designed as a portfolio-quality AI Engineering project demonstrating:

- Retrieval Engineering
- Hybrid Search (Dense + Sparse)
- Cross-Encoder Reranking
- Prompt Engineering
- LLMOps and evaluation
- Dockerized, modular Python architecture

## Features

- Official-source-only ingestion
- HTML / PDF / TXT preprocessing
- Recursive chunking with metadata
- Dense retrieval using BAAI/bge-small-en-v1.5
- Sparse retrieval using BM25
- Hybrid retrieval and reranking
- Groq-powered grounded answer generation
- Source citations and confidence scores
- Streaming CLI responses
- Benchmark suite with pass/fail evaluation
- Dockerized local execution

## Architecture

User Query  
→ Language Detection  
→ Query Expansion  
→ Hybrid Retrieval (FAISS + BM25)  
→ Cross-Encoder Reranking  
→ Prompt Construction  
→ Groq LLM  
→ Citation + Confidence Output

## Tech Stack

- Python 3.11
- PyMuPDF
- BeautifulSoup
- SentenceTransformers
- FAISS
- rank-bm25
- Groq API
- MarianMT / Transformers
- Typer + Rich
- Loguru
- Pytest
- Docker

## Project Structure

```text
config/        configuration and prompt templates
data/          raw, cleaned, processed, and benchmark data
src/           application source code
tests/         unit tests
indexes/       FAISS and BM25 indexes
embeddings/    cached embeddings
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Create environment file

```bash
cp .env.example .env
```

Set your Groq API key inside `.env`.

### 3. Install dependencies

```bash
make install
make init
```

## Run pipeline

### Ingest official documents

```bash
make ingest
```

### Preprocess documents

```bash
make preprocess
```

### Chunk documents

```bash
make chunk
```

### Build indexes

```bash
make index
```

### Run chatbot

```bash
make chat
```

### Run benchmarks

```bash
make benchmark
```

## Example CLI

```text
Ask > What are annual leave rules?

Answer:
Employees are entitled to annual leave after a qualifying period...[8]

Confidence: 0.84 (High)
Sources:
1. Employment Ordinance — Page 17
```

## Benchmarking

Benchmark cases are stored in:

```text
data/benchmarks/benchmark_cases.jsonl
```

Benchmark results are written to:

```text
data/benchmarks/benchmark_results.jsonl
```

Each case evaluates:

- expected document retrieval
- expected keyword presence
- fallback behavior
- latency compliance
- confidence score

## Docker

### Run chatbot

```bash
docker compose up pobot
```

### Run ingestion

```bash
docker compose up pobot-ingest
```

### Run benchmark

```bash
docker compose up pobot-benchmark
```

## Testing

```bash
make test
```

## Future Improvements

- Web UI / FastAPI interface
- Better multilingual translation coverage
- Citation span validation
- More robust legal benchmark dataset
- Structured answer schema
- Observability dashboard for latency and retrieval quality

## Disclaimer

This project is for informational and engineering demonstration purposes only. It is not legal advice. Users should verify important answers against official Hong Kong government sources.
.PHONY: help install test lint format clean init ingest preprocess chunk index chat benchmark all

PYTHON := python
PIP := pip

help:
	@echo "PoBot RAG commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make init         - Create required directories"
	@echo "  make test         - Run pytest"
	@echo "  make lint         - Run code quality checks"
	@echo "  make format       - Format code"
	@echo "  make ingest       - Collect raw official documents"
	@echo "  make preprocess   - Clean and normalize documents"
	@echo "  make chunk        - Build chunk corpus"
	@echo "  make index        - Build embeddings + FAISS + BM25"
	@echo "  make chat         - Run CLI chatbot"
	@echo "  make benchmark    - Run benchmark suite"
	@echo "  make clean        - Remove caches and generated artifacts"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

init:
	mkdir -p data/raw data/cleaned data/processed
	mkdir -p embeddings indexes/faiss indexes/bm25 logs data/benchmarks
	touch data/raw/.gitkeep data/cleaned/.gitkeep data/processed/.gitkeep
	touch embeddings/.gitkeep indexes/faiss/.gitkeep indexes/bm25/.gitkeep

test:
	pytest tests/ -v

lint:
	python -m compileall config src tests

format:
	@echo "Add black/ruff later if desired; compile check used for now."

ingest:
	$(PYTHON) -m src.ingestion.collect

preprocess:
	$(PYTHON) -m src.ingestion.preprocess

chunk:
	$(PYTHON) -m src.ingestion.chunk

index:
	$(PYTHON) -m src.retrieval.build_indexes

chat:
	$(PYTHON) src/main.py chat

benchmark:
	$(PYTHON) src/main.py benchmark

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage .coverage.*
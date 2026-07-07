#!/usr/bin/env bash
# scripts/init_project.sh
set -e

echo "🏗  Creating PoBot RAG directory structure…"

dirs=(
  "config"
  "data/raw" "data/cleaned" "data/processed"
  "embeddings"
  "indexes/faiss" "indexes/bm25"
  "src/ingestion" "src/embedding" "src/retrieval"
  "src/llm" "src/chatbot" "src/evaluation"
  "src/translation" "src/utils"
  "tests/config" "tests/ingestion" "tests/retrieval"
  "tests/llm" "tests/utils"
  "logs"
)

for dir in "${dirs[@]}"; do
  mkdir -p "$dir"
  touch "$dir/__init__.py" 2>/dev/null || true
done

# Package markers for top-level modules
touch config/__init__.py src/__init__.py tests/__init__.py

# Placeholder metadata file
echo '[]' > data/metadata.json

echo "✅ Project structure ready."
echo "Next: cp .env.example .env && edit .env with your GROQ_API_KEY"
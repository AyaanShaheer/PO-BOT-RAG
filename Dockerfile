# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="PoBot RAG"
LABEL org.opencontainers.image.description="AI Assistant for Hong Kong Labour Regulations"

WORKDIR /app

# Non-root user for security
RUN groupadd --gid 1001 pobot && useradd --uid 1001 --gid pobot --no-create-home pobot

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=pobot:pobot . .

# Create runtime data dirs (owned by app user)
RUN mkdir -p data/raw data/cleaned data/processed embeddings indexes/faiss indexes/bm25 logs \
    && chown -R pobot:pobot data embeddings indexes logs

USER pobot

# Default env — override at runtime with --env-file or -e flags
ENV APP_ENV=prod \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "src/main.py"]
CMD ["chat"]
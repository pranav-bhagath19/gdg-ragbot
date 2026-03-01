# ─── Dockerfile ───────────────────────────────────────────────────────────────
# Production-ready Docker image for the RAG Chatbot API
# Pre-downloads the sentence-transformers model to avoid cold start delays

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── PRE-DOWNLOAD EMBEDDING MODEL ──────────────────────────────────────────────
# This bakes the ~90MB model into the image so the first request doesn't
# trigger a download (which would take 10-15s and violate our <4s target).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model pre-downloaded ✅')"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/data /app/logs

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STORAGE_PATH=/app/data

# Expose port
EXPOSE 8000

# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── START ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

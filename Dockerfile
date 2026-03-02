# ─── Dockerfile ───────────────────────────────────────────────────────────────
# Production-ready Docker image for the RAG Chatbot API
# Optimized for size: CPU-only PyTorch, no CUDA (~3.5GB instead of 6.5GB)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies, build wheels, then remove build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~1.5GB vs full torch with CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Remove build tools to save space
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ── PRE-DOWNLOAD EMBEDDING MODEL ──────────────────────────────────────────────
# Bakes the ~90MB model into the image to avoid cold start download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model pre-downloaded')"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/data /app/logs

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STORAGE_PATH=/app/data
# Model is baked into the image — prevent network calls on startup
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Expose port (Railway overrides this with $PORT)
EXPOSE 8000

# No Docker HEALTHCHECK — Railway handles healthchecks via healthcheckPath in railway.toml

# ── START ─────────────────────────────────────────────────────────────────────
# Railway overrides this with startCommand from railway.toml (uses $PORT)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

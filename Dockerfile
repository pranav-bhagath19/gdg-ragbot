# ─── Dockerfile ───────────────────────────────────────────────────────────────
# Production Docker image for the RAG Chatbot API
# Uses fastembed (ONNX Runtime) instead of PyTorch — ~1GB image instead of 3.5GB

FROM python:3.11-slim

WORKDIR /app

# Copy and install Python dependencies (no torch = fast + small)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── PRE-DOWNLOAD EMBEDDING MODEL ──────────────────────────────────────────────
# Bakes the ~90MB ONNX model into the image so first request is fast
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'); print('Model pre-downloaded')"

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/data /app/logs

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STORAGE_PATH=/app/data

# Expose port (Railway overrides this with $PORT)
EXPOSE 8000

# ── START ─────────────────────────────────────────────────────────────────────
# Use shell form so $PORT is expanded at runtime (Render/Railway set PORT dynamically)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1

"""
main.py - FastAPI application entry point.
Place this file at: rag-chatbot/main.py

Startup actions:
1. Pre-download / load sentence-transformers model (avoids cold start delay)
2. Initialize hnswlib vector store (loads from disk if exists)
3. Register all routers
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys

from config import get_settings
from routers import ingest, chat, documents, appwrite_ingest
from services import embed, vectorstore

# ─── Logging Configuration ────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    backtrace=True,
    diagnose=True,
)


# ─── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load embedding model + initialize vector store.
    This prevents cold start delays on the first request.
    """
    settings = get_settings()
    start = time.time()

    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)

    # 1. Pre-load embedding model
    logger.info("📦 Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    try:
        embed.load_model()
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise

    # 2. Initialize vector store
    logger.info(f"📁 Initializing vector store at: {settings.storage_path}")
    try:
        vectorstore.init_store(settings.storage_path)
        stats = vectorstore.get_stats()
        logger.info(f"✅ Vector store ready: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
    except Exception as e:
        logger.error(f"❌ Failed to initialize vector store: {e}")
        raise

    elapsed = time.time() - start
    logger.info(f"✅ App started in {elapsed:.1f}s — Ready to serve requests")
    logger.info("=" * 60)

    yield  # App is running

    logger.info("🛑 Shutting down...")


# ─── App Instance ─────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Free-Tier RAG Chatbot API

A Retrieval-Augmented Generation (RAG) API that lets you chat with your documents.

### Features
- 📄 Upload PDF, Excel (XLSX), and Word (DOCX) documents
- 🔍 Semantic search using sentence-transformers embeddings
- 🤖 LLM-powered answers via Groq (Llama 3.1 8B) with Gemini Flash fallback
- 📚 Source citations with document and page references
- 💾 Persistent vector storage using hnswlib

### Quick Start
1. `POST /ingest` — Upload a document
2. `POST /chat` — Ask questions about it
3. `GET /documents` — List ingested documents
    """,
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global Error Handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)[:500]},
    )


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
async def health_check():
    """
    Health check endpoint. Used by UptimeRobot for cold-start prevention.
    Returns system status and vector store stats.
    """
    stats = vectorstore.get_stats()
    model_loaded = embed._model is not None
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "model_loaded": model_loaded,
        "vector_store": stats,
        "providers": {
            "groq_configured": bool(settings.groq_api_key),
            "gemini_configured": bool(settings.gemini_api_key),
        }
    }


@app.get("/", tags=["System"], summary="Root / Welcome")
async def root():
    """Welcome endpoint with API documentation link."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/health",
        "version": settings.app_version,
    }


# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(appwrite_ingest.router, tags=["Appwrite Ingestion"])


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level="info",
    )

"""
routers/chat.py - POST /chat endpoint for RAG-powered chat.
Place this file at: rag-chatbot/routers/chat.py

Pipeline: embed query → ANN search → build prompt → call LLM (Groq/Gemini) → return JSON
"""
import time
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from loguru import logger

from config import get_settings, Settings
from services.retriever import retrieve, format_context
from services.prompt import build_prompt, SYSTEM_PROMPT
from services.llm import call_llm_with_failover
from services import vectorstore

router = APIRouter()


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional conversation history (last N turns)"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve"
    )


class Citation(BaseModel):
    source_num: int
    document: str
    page: str
    chunk_index: int
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    provider: str
    model: str
    fallback_used: bool
    latency_ms: int
    chunks_retrieved: int
    query: str


# ─── Rate limiting (in-memory, per-process) ────────────────────────────────────

from collections import deque
from datetime import datetime, timedelta

_request_times: deque = deque()
_rate_limit_lock = None

def _check_rate_limit(requests_per_minute: int) -> bool:
    """Simple in-memory rate limiter. Returns True if request is allowed."""
    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=1)

    # Remove old entries
    while _request_times and _request_times[0] < cutoff:
        _request_times.popleft()

    if len(_request_times) >= requests_per_minute:
        return False

    _request_times.append(now)
    return True


# ─── Chat Endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Chat with your documents")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    """
    RAG-powered chat endpoint.

    1. Embeds your query using sentence-transformers
    2. Retrieves top-k relevant document chunks from hnswlib
    3. Builds a context-aware prompt
    4. Calls Groq (Llama 3.1 8B) with automatic Gemini Flash fallback
    5. Returns answer + source citations

    Target latency: < 4 seconds (warm invocation)
    """
    total_start = time.time()

    # Rate limit guard
    if not _check_rate_limit(settings.rate_limit_per_minute):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {settings.rate_limit_per_minute} requests/minute"
        )

    # Check if we have any documents
    stats = vectorstore.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents ingested yet. Please upload documents first via POST /ingest."
        )

    # Step 1: Retrieve relevant chunks
    top_k = request.top_k or settings.top_k_chunks
    try:
        chunks = await retrieve(request.query, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    if not chunks:
        return ChatResponse(
            answer="I couldn't find any relevant information in the documents to answer your question.",
            citations=[],
            provider="none",
            model="none",
            fallback_used=False,
            latency_ms=round((time.time() - total_start) * 1000),
            chunks_retrieved=0,
            query=request.query,
        )

    # Step 2: Format context and citations
    context, citations = format_context(chunks)

    # Step 3: Build prompt
    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    messages = build_prompt(request.query, context, history)

    # Step 4: Call LLM with failover
    try:
        answer, usage = await call_llm_with_failover(messages, SYSTEM_PROMPT)
    except RuntimeError as e:
        logger.error(f"All LLM providers failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unavailable: {str(e)[:300]}"
        )
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)[:300]}")

    total_ms = round((time.time() - total_start) * 1000)
    logger.info(f"Chat complete | provider={usage['provider']} | total={total_ms}ms | fallback={usage.get('fallback_used', False)}")

    return ChatResponse(
        answer=answer,
        citations=[Citation(**c) for c in citations],
        provider=usage.get("provider", "unknown"),
        model=usage.get("model", "unknown"),
        fallback_used=usage.get("fallback_used", False),
        latency_ms=total_ms,
        chunks_retrieved=len(chunks),
        query=request.query,
    )

"""
routers/chat.py - POST /chat endpoint.
"""
import time
from typing import List, Optional, Dict
from collections import deque
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from loguru import logger

from config import get_settings, Settings
from services.retriever import retrieve, format_context
from services.prompt import build_prompt, SYSTEM_PROMPT
from services.llm import call_llm_with_failover
from services import vectorstore

router = APIRouter()


# ─── Models ───────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatMessage]] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    model: str
    fallback_used: bool
    latency_ms: int


# ─── Rate limiter ──────────────────────────────────────────────────────────────

_request_times: deque = deque()

def _check_rate_limit(requests_per_minute: int) -> bool:
    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=1)
    while _request_times and _request_times[0] < cutoff:
        _request_times.popleft()
    if len(_request_times) >= requests_per_minute:
        return False
    _request_times.append(now)
    return True


# ─── Friendly error messages (no technical details) ───────────────────────────

FALLBACK_ANSWER = "I'm sorry, I couldn't find relevant information to answer that question."
NO_DOCS_ANSWER  = "I don't have any documents to search through yet. Please upload some documents first."
RATE_ANSWER     = "I'm receiving too many requests right now. Please try again in a moment."
ERROR_ANSWER    = "I wasn't able to process that request. Please try again."


# ─── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Chat with your documents")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
):
    total_start = time.time()

    # Rate limit — return friendly message, not error code
    if not _check_rate_limit(settings.rate_limit_per_minute):
        logger.warning("Rate limit hit")
        return ChatResponse(
            answer=RATE_ANSWER,
            provider="none",
            model="none",
            fallback_used=False,
            latency_ms=0,
        )

    # No documents — return friendly message
    stats = vectorstore.get_stats()
    if stats["total_chunks"] == 0:
        logger.warning("Chat called with empty vector store")
        return ChatResponse(
            answer=NO_DOCS_ANSWER,
            provider="none",
            model="none",
            fallback_used=False,
            latency_ms=0,
        )

    # Retrieve relevant chunks
    top_k = request.top_k or settings.top_k_chunks
    try:
        chunks = await retrieve(request.query, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return ChatResponse(
            answer=FALLBACK_ANSWER,
            provider="none",
            model="none",
            fallback_used=False,
            latency_ms=round((time.time() - total_start) * 1000),
        )

    # No relevant chunks found
    if not chunks:
        return ChatResponse(
            answer=FALLBACK_ANSWER,
            provider="none",
            model="none",
            fallback_used=False,
            latency_ms=round((time.time() - total_start) * 1000),
        )

    # Format context (internal use only — never shown to user)
    context, citations = format_context(chunks)
    logger.debug(f"Citations (internal): {citations}")

    # Build prompt
    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    messages = build_prompt(request.query, context, history)

    # Call LLM with full 4-key fallback chain
    try:
        answer, usage = await call_llm_with_failover(messages, SYSTEM_PROMPT)
    except RuntimeError as e:
        # All 4 keys failed — log internally, show friendly message to user
        logger.error(f"All LLM providers failed: {e}")
        return ChatResponse(
            answer=FALLBACK_ANSWER,
            provider="none",
            model="none",
            fallback_used=True,
            latency_ms=round((time.time() - total_start) * 1000),
        )
    except Exception as e:
        logger.error(f"Unexpected LLM error: {e}")
        return ChatResponse(
            answer=FALLBACK_ANSWER,
            provider="none",
            model="none",
            fallback_used=True,
            latency_ms=round((time.time() - total_start) * 1000),
        )

    total_ms = round((time.time() - total_start) * 1000)
    logger.info(
        f"Chat complete | provider={usage.get('key_used')} | "
        f"total={total_ms}ms | fallback={usage.get('fallback_used')}"
    )

    return ChatResponse(
        answer=answer,
        provider=usage.get("provider", "unknown"),
        model=usage.get("model", "unknown"),
        fallback_used=usage.get("fallback_used", False),
        latency_ms=total_ms,
    )
"""
services/llm.py - LLM client with Groq primary and Gemini Flash automatic failover.
Place this file at: rag-chatbot/services/llm.py

Failover logic:
1. Try Groq API
2. If Groq fails (any error) → retry once after 1s
3. If retry fails → automatically switch to Gemini Flash
4. Log which provider was used
5. Return identical response format regardless of provider
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from config import get_settings


# ─── Groq Client ──────────────────────────────────────────────────────────────

async def call_groq(
    messages: List[Dict[str, str]],
    system_prompt: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call Groq API with Llama 3.1 8B.
    Returns (answer_text, usage_info).
    Raises exception on any failure.
    """
    settings = get_settings()

    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY not configured")

    from groq import Groq, APIError, RateLimitError

    client = Groq(api_key=settings.groq_api_key)

    all_messages = [{"role": "system", "content": system_prompt}] + messages

    start_time = time.time()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=all_messages,
        max_tokens=settings.max_tokens,
        temperature=0.1,
    )
    elapsed = time.time() - start_time

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("Groq returned empty response")

    usage = {
        "provider": "groq",
        "model": settings.groq_model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "latency_ms": round(elapsed * 1000),
    }

    logger.info(f"Groq responded in {elapsed:.2f}s | tokens: {response.usage.total_tokens}")
    return answer.strip(), usage


# ─── Gemini Client ─────────────────────────────────────────────────────────────

async def call_gemini(
    messages: List[Dict[str, str]],
    system_prompt: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call Google Gemini Flash API.
    Returns (answer_text, usage_info).
    Raises exception on any failure.
    """
    settings = get_settings()

    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_prompt,
    )

    # Convert OpenAI-style messages to Gemini format
    gemini_history = []
    current_user_message = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            current_user_message = content
        elif role == "assistant":
            if current_user_message:
                gemini_history.append({"role": "user", "parts": [current_user_message]})
                gemini_history.append({"role": "model", "parts": [content]})
                current_user_message = None

    # The last user message is the actual prompt
    last_user_msg = current_user_message or messages[-1]["content"]

    start_time = time.time()

    # Start chat with history if any
    if gemini_history:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(last_user_msg)
    else:
        response = model.generate_content(last_user_msg)

    elapsed = time.time() - start_time

    answer = response.text
    if not answer or not answer.strip():
        raise ValueError("Gemini returned empty response")

    usage = {
        "provider": "gemini",
        "model": settings.gemini_model,
        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "latency_ms": round(elapsed * 1000),
    }

    logger.info(f"Gemini responded in {elapsed:.2f}s")
    return answer.strip(), usage


# ─── Failover Manager ──────────────────────────────────────────────────────────

async def call_llm_with_failover(
    messages: List[Dict[str, str]],
    system_prompt: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Primary LLM router with automatic Groq → Gemini failover.

    Flow:
    1. Try Groq
    2. If Groq fails → wait 1s → retry Groq once
    3. If retry fails → switch to Gemini Flash
    4. Log which provider was used
    5. Return (answer, usage_info) — same format regardless of provider

    Raises RuntimeError only if BOTH providers fail.
    """
    groq_errors = []

    # ── Attempt 1: Groq ─────────────────────────────────────────────────────
    logger.debug("Attempting Groq API call...")
    try:
        answer, usage = await call_groq(messages, system_prompt)
        usage["fallback_used"] = False
        logger.info(f"✅ LLM provider: GROQ | latency: {usage['latency_ms']}ms")
        return answer, usage
    except Exception as e:
        groq_errors.append(f"Groq attempt 1: {type(e).__name__}: {str(e)[:200]}")
        logger.warning(f"Groq attempt 1 failed: {type(e).__name__}: {str(e)[:200]}")

    # ── Retry: Groq (after 1s delay) ────────────────────────────────────────
    logger.debug("Retrying Groq API call after 1s...")
    await asyncio.sleep(1)
    try:
        answer, usage = await call_groq(messages, system_prompt)
        usage["fallback_used"] = False
        usage["retry_used"] = True
        logger.info(f"✅ LLM provider: GROQ (retry) | latency: {usage['latency_ms']}ms")
        return answer, usage
    except Exception as e:
        groq_errors.append(f"Groq attempt 2: {type(e).__name__}: {str(e)[:200]}")
        logger.warning(f"Groq attempt 2 failed: {type(e).__name__}: {str(e)[:200]}")

    # ── Fallback: Gemini Flash ───────────────────────────────────────────────
    logger.info("🔄 Falling back to Gemini Flash...")
    try:
        answer, usage = await call_gemini(messages, system_prompt)
        usage["fallback_used"] = True
        usage["groq_errors"] = groq_errors
        logger.info(f"✅ LLM provider: GEMINI FLASH (fallback) | latency: {usage['latency_ms']}ms")
        return answer, usage
    except Exception as e:
        gemini_error = f"Gemini: {type(e).__name__}: {str(e)[:200]}"
        logger.error(f"Gemini fallback also failed: {gemini_error}")
        raise RuntimeError(
            f"All LLM providers failed.\n"
            f"Groq errors: {groq_errors}\n"
            f"Gemini error: {gemini_error}"
        )

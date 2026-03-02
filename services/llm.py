"""
services/llm.py - LLM client with 4-key rollback chain.

Attempt order:
1. Groq KEY 1        (GROQ_API_KEY)
2. Groq KEY 2        (GROQ_API_KEY_BACKUP)
3. Gemini KEY 1      (GEMINI_API_KEY)
4. Gemini KEY 2      (GEMINI_API_KEY_BACKUP)

If all 4 fail → returns error to user.
"""
import asyncio
import time
from typing import List, Dict, Tuple
from loguru import logger

from config import get_settings


# ─── Groq caller ──────────────────────────────────────────────────────────────

async def call_groq(
    messages: List[Dict[str, str]],
    system_prompt: str,
    api_key: str,
    key_label: str,
) -> Tuple[str, Dict]:
    settings = get_settings()

    if not api_key or not api_key.strip():
        raise ValueError(f"Groq {key_label} key is empty")

    from groq import Groq

    client = Groq(api_key=api_key.strip())
    all_messages = [{"role": "system", "content": system_prompt}] + messages

    start = time.time()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=all_messages,
        max_tokens=settings.max_tokens,
        temperature=0.1,
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("Groq returned empty response")

    logger.info(f"✅ Groq {key_label} key responded in {elapsed:.2f}s")

    return answer.strip(), {
        "provider": "groq",
        "key_used": f"groq_{key_label}",
        "model": settings.groq_model,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "latency_ms": round(elapsed * 1000),
        "fallback_used": False,
    }


# ─── Gemini caller ─────────────────────────────────────────────────────────────

async def call_gemini(
    messages: List[Dict[str, str]],
    system_prompt: str,
    api_key: str,
    key_label: str,
) -> Tuple[str, Dict]:
    settings = get_settings()

    if not api_key or not api_key.strip():
        raise ValueError(f"Gemini {key_label} key is empty")

    import google.generativeai as genai

    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_prompt,
    )

    # Convert to Gemini format
    gemini_history = []
    current_user_msg = None

    for msg in messages:
        if msg["role"] == "user":
            current_user_msg = msg["content"]
        elif msg["role"] == "assistant" and current_user_msg:
            gemini_history.append({"role": "user", "parts": [current_user_msg]})
            gemini_history.append({"role": "model", "parts": [msg["content"]]})
            current_user_msg = None

    last_msg = current_user_msg or messages[-1]["content"]

    start = time.time()
    if gemini_history:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(last_msg)
    else:
        response = model.generate_content(last_msg)
    elapsed = time.time() - start

    answer = response.text
    if not answer or not answer.strip():
        raise ValueError("Gemini returned empty response")

    logger.info(f"✅ Gemini {key_label} key responded in {elapsed:.2f}s")

    return answer.strip(), {
        "provider": "gemini",
        "key_used": f"gemini_{key_label}",
        "model": settings.gemini_model,
        "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "latency_ms": round(elapsed * 1000),
        "fallback_used": True,
    }


# ─── Main failover chain ───────────────────────────────────────────────────────

async def call_llm_with_failover(
    messages: List[Dict[str, str]],
    system_prompt: str,
) -> Tuple[str, Dict]:
    """
    Tries all 4 keys in order:
    1. Groq key 1
    2. Groq key 2
    3. Gemini key 1
    4. Gemini key 2
    """
    settings = get_settings()
    all_errors = []

    # Build the attempt chain
    attempts = [
        ("groq",   "key1",  settings.groq_api_key),
        ("groq",   "key2",  settings.groq_api_key_backup),
        ("gemini", "key1",  settings.gemini_api_key),
        ("gemini", "key2",  settings.gemini_api_key_backup),
    ]

    for provider, key_label, api_key in attempts:

        # Skip if key not configured
        if not api_key or not api_key.strip():
            logger.debug(f"Skipping {provider} {key_label} — not configured")
            continue

        logger.info(f"Trying {provider.upper()} {key_label}...")

        try:
            if provider == "groq":
                answer, usage = await call_groq(messages, system_prompt, api_key, key_label)
            else:
                answer, usage = await call_gemini(messages, system_prompt, api_key, key_label)

            # Success — log which key worked and return
            logger.info(f"✅ SUCCESS — {provider.upper()} {key_label} | latency: {usage['latency_ms']}ms")
            usage["attempt"] = f"{provider}_{key_label}"
            return answer, usage

        except Exception as e:
            err = f"{provider} {key_label}: {type(e).__name__}: {str(e)[:150]}"
            all_errors.append(err)
            logger.warning(f"❌ Failed — {err}")

            # Small delay before next attempt
            await asyncio.sleep(1)

    # All 4 failed
    error_summary = "\n".join(f"  - {e}" for e in all_errors)
    raise RuntimeError(f"All 4 API keys failed:\n{error_summary}")
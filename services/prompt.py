"""
services/prompt.py - Prompt builder.
"""
from typing import List, Dict, Optional


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions strictly from the provided document content.

Rules:
- Answer ONLY using the information given to you below.
- If the information is not present, say "I don't have information on that."
- NEVER mention sources, documents, page numbers, or citations.
- NEVER say "Based on the provided context" or "According to the documents".
- NEVER reveal technical details like API limits, rate limits, keys, or errors.
- NEVER say things like "I was unable to retrieve" or "the system encountered".
- Write in clean, natural, direct sentences as if you simply know the answer.
- Do not add any disclaimers, footnotes, or references of any kind.
- Just answer the question directly and naturally.
"""


def build_prompt(
    query: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages = []

    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_message = f"""Information:
---
{context}
---

Question: {query}"""

    messages.append({"role": "user", "content": user_message})
    return messages
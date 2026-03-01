"""
services/prompt.py - Builds context-aware prompts for the LLM.
Place this file at: rag-chatbot/services/prompt.py
"""
from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided document context.

Instructions:
- Answer ONLY from the provided context below.
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
- Cite the source numbers (e.g., [Source 1], [Source 2]) when referencing information.
- Be concise, accurate, and helpful.
- Do not make up information or use external knowledge beyond what's provided.
"""


def build_prompt(
    query: str,
    context: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build the messages list for the LLM API call.

    Args:
        query: The user's question
        context: Formatted retrieval context from retrieved chunks
        history: Optional list of previous turns [{role, content}, ...]

    Returns:
        List of message dicts for the LLM API.
    """
    messages = []

    # Add conversation history (limit to last 6 turns to stay within token budget)
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Add the current query with context
    user_message = f"""Context from documents:
---
{context}
---

Question: {query}

Please answer based on the context above. Cite source numbers where relevant."""

    messages.append({"role": "user", "content": user_message})

    return messages

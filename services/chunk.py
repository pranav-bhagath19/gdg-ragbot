"""
services/chunk.py - Text chunking with recursive character splitting.
Place this file at: rag-chatbot/services/chunk.py
"""
from typing import List, Dict, Any
from loguru import logger


class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks of approximately `chunk_size` tokens
    with `overlap` tokens of context carryover between chunks.
    Uses word-boundary-aware splitting.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Approximate: 1 token ≈ 4 characters (rough estimate for English)
        self.chars_per_token = 4

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.chars_per_token

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting word boundaries."""
        max_chars = self.chunk_size * self.chars_per_token
        overlap_chars = self.chunk_overlap * self.chars_per_token

        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chars

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Find a good break point (sentence or word boundary)
            # Try to break at sentence end first
            break_pos = text.rfind('. ', start, end)
            if break_pos == -1 or break_pos <= start:
                # Fall back to word boundary
                break_pos = text.rfind(' ', start, end)
            if break_pos == -1 or break_pos <= start:
                # No good break point, force split
                break_pos = end

            chunk = text[start:break_pos + 1].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = max(start + 1, break_pos + 1 - overlap_chars)

        return [c for c in chunks if c.strip()]

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of document dicts into chunks.
        Each input doc: {text, page, source, type}
        Each output chunk adds: {chunk_index, total_chunks}
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue

            text_chunks = self._split_text(text)
            total = len(text_chunks)

            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "page": doc.get("page", 1),
                    "source": doc.get("source", "unknown"),
                    "type": doc.get("type", "unknown"),
                    "chunk_index": i,
                    "total_chunks": total,
                })

        logger.debug(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

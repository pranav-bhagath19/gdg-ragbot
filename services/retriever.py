"""
services/retriever.py - Retrieval pipeline: embed query → ANN search → return context.
Place this file at: rag-chatbot/services/retriever.py
"""
from typing import List, Dict, Any, Tuple
from loguru import logger

from services.embed import embed_query
from services import vectorstore


async def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Embed user query (~15ms)
    2. ANN search in hnswlib (~5ms)
    3. Return top-k relevant chunks with metadata

    Returns list of chunk dicts with 'score' field.
    """
    logger.debug(f"Retrieving top-{top_k} chunks for query: '{query[:80]}...'")

    query_vector = embed_query(query)
    results = await vectorstore.search(query_vector, top_k=top_k)

    logger.debug(f"Retrieved {len(results)} chunks. Top score: {results[0]['score']:.3f}" if results else "No chunks retrieved")
    return results


def format_context(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Returns (context_string, citations_list).
    """
    if not chunks:
        return "No relevant context found.", []

    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        score = chunk.get("score", 0)

        context_parts.append(
            f"[Source {i}] {source} (page/section: {page})\n{text}"
        )

        citations.append({
            "source_num": i,
            "document": source,
            "page": str(page),
            "chunk_index": chunk.get("chunk_index", 0),
            "relevance_score": round(score, 3),
        })

    context_string = "\n\n---\n\n".join(context_parts)
    return context_string, citations

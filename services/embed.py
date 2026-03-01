"""
services/embed.py - Embedding service using sentence-transformers.
Place this file at: rag-chatbot/services/embed.py

The model is loaded ONCE at module import (preloaded).
This prevents per-request model loading (~10-15s delay).
"""
from typing import List
import numpy as np
from loguru import logger

# Module-level singleton — loaded once at startup
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_model():
    """Load the sentence-transformers model. Called once at app startup."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME} ...")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Embedding model loaded successfully. Dim={EMBEDDING_DIM}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _model


def get_model():
    """Get the loaded model, loading it if necessary."""
    global _model
    if _model is None:
        return load_model()
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of text strings.
    Returns numpy array of shape (len(texts), EMBEDDING_DIM).
    ~15ms per chunk on CPU.
    """
    if not texts:
        return np.array([]).reshape(0, EMBEDDING_DIM)

    model = get_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns numpy array of shape (EMBEDDING_DIM,).
    """
    result = embed_texts([query])
    return result[0]

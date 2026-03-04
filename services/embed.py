"""
services/embed.py - Embedding service using fastembed (ONNX Runtime).
Place this file at: rag-chatbot/services/embed.py

Uses ONNX Runtime instead of PyTorch — ~5x smaller image, ~3x less RAM.
The model is loaded LAZILY on first request to keep startup fast.
"""
from typing import List
import numpy as np
from loguru import logger

# Module-level singleton — loaded on first request
_model = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_model():
    """Load the fastembed model. Called lazily on first embed request."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME} ...")
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL_NAME)
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


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of text strings in batches to avoid OOM.
    Returns numpy array of shape (len(texts), EMBEDDING_DIM).
    """
    if not texts:
        return np.array([]).reshape(0, EMBEDDING_DIM)

    model = get_model()
    try:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} ({len(batch)} texts)")
            batch_embeddings = list(model.embed(batch))
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings, dtype=np.float32)
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

"""
services/vectorstore.py - numpy-based vector store with disk persistence.
Place this file at: rag-chatbot/services/vectorstore.py

Features:
- Stores vectors in a simple numpy array (no C++ build tools required!)
- Persists index + metadata to disk (JSON + npy)
- Supports add, search, delete, list
- Thread-safe with asyncio.Lock
"""
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from services.embed import EMBEDDING_DIM

# Concurrency lock
_index_lock = asyncio.Lock()

# Module-level globals (singleton)
_vectors: Optional[np.ndarray] = None
_metadata: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata dict
_doc_registry: Dict[str, Dict[str, Any]] = {}  # doc_id -> doc info
_chunk_ids: List[str] = []  # Maps row index in _vectors to chunk_id

# File paths
_vectors_path: Optional[Path] = None
_meta_path: Optional[Path] = None
_registry_path: Optional[Path] = None


def init_store(storage_path: str):
    """
    Initialize the vector store. Call once at app startup.
    Loads existing index from disk if available.
    """
    global _vectors, _vectors_path, _meta_path, _registry_path
    global _metadata, _doc_registry, _chunk_ids

    storage = Path(storage_path)
    storage.mkdir(parents=True, exist_ok=True)

    _vectors_path = storage / "vectors.npy"
    _meta_path = storage / "metadata.json"
    _registry_path = storage / "registry.json"

    # Load metadata from disk
    if _meta_path.exists():
        with open(_meta_path, "r") as f:
            saved = json.load(f)
            _metadata = saved.get("metadata", {})
            _chunk_ids = saved.get("chunk_ids", [])
        logger.info(f"Loaded metadata: {len(_metadata)} chunks")

    # Load document registry
    if _registry_path.exists():
        with open(_registry_path, "r") as f:
            _doc_registry = json.load(f)
        logger.info(f"Loaded registry: {len(_doc_registry)} documents")

    # Load vectors
    if _vectors_path.exists() and len(_chunk_ids) > 0:
        _vectors = np.load(str(_vectors_path))
        logger.info(f"Loaded vectors from disk: {_vectors.shape[0]} vectors")
    else:
        _vectors = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        logger.info("Created new empty vector index")


def _persist():
    """Save index and metadata to disk."""
    if _vectors is None or _vectors_path is None:
        return

    # Save numpy array
    np.save(str(_vectors_path), _vectors)

    # Save metadata
    with open(_meta_path, "w") as f:
        json.dump({
            "metadata": _metadata,
            "chunk_ids": _chunk_ids,
        }, f, indent=2)

    # Save registry
    with open(_registry_path, "w") as f:
        json.dump(_doc_registry, f, indent=2)

    logger.debug("Vector store persisted to disk")


async def add_chunks(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    doc_id: str,
    doc_info: Dict[str, Any],
) -> int:
    """
    Add a batch of chunk embeddings + metadata to the index.
    """
    global _vectors, _chunk_ids

    async with _index_lock:
        if _vectors is None:
            raise RuntimeError("Vector store not initialized. Call init_store() first.")

        # Normalize embeddings for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        normalized_embeddings = embeddings / norms

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            _chunk_ids.append(chunk_id)
            _metadata[chunk_id] = {
                **chunk,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
            }

        # Append vectors
        _vectors = np.vstack([_vectors, normalized_embeddings])

        # Register the document
        _doc_registry[doc_id] = {
            **doc_info,
            "chunk_count": len(chunks),
        }

        _persist()
        logger.info(f"Added {len(chunks)} chunks for doc_id={doc_id}")
        return len(chunks)


async def search(
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search for top-k most similar chunks.
    """
    if _vectors is None or len(_vectors) == 0:
        logger.warning("Search called on empty index")
        return []

    # Normalize query embedding
    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        q_norm = 1e-10
    q_vec = query_embedding / q_norm

    # Cosine similarity is just dot product for normalized vectors
    similarities = np.dot(_vectors, q_vec.T).flatten()
    
    # Get top k indices
    k = min(top_k, len(similarities))
    # argpartition is faster than argsort for top k
    if k < len(similarities):
        top_indices = np.argpartition(similarities, -k)[-k:]
    else:
        top_indices = np.arange(len(similarities))
        
    # Sort the top k indices by similarity descending
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        chunk_id = _chunk_ids[idx]
        if chunk_id in _metadata:
            meta = dict(_metadata[chunk_id])
            meta["score"] = float(similarities[idx])
            results.append(meta)

    return results


async def delete_document(doc_id: str) -> bool:
    """
    Delete all chunks belonging to a document.
    """
    global _vectors, _chunk_ids
    
    async with _index_lock:
        if doc_id not in _doc_registry:
            return False

        # Find indices to remove
        indices_to_remove = set()
        for idx, chunk_id in enumerate(_chunk_ids):
            if _metadata[chunk_id].get("doc_id") == doc_id:
                indices_to_remove.add(idx)
        
        if not indices_to_remove:
             # Just clean up the registry if chunks are missing somehow
             del _doc_registry[doc_id]
             _persist()
             return True

        indices_to_keep = [i for i in range(len(_chunk_ids)) if i not in indices_to_remove]
        
        # Update numpy array
        _vectors = _vectors[indices_to_keep]
        
        # Update maps
        new_chunk_ids = []
        for idx in indices_to_keep:
            new_chunk_ids.append(_chunk_ids[idx])
            
        for idx in indices_to_remove:
            chunk_id = _chunk_ids[idx]
            del _metadata[chunk_id]
            
        _chunk_ids = new_chunk_ids
        del _doc_registry[doc_id]
        
        _persist()
        logger.info(f"Deleted document {doc_id}: removed {len(indices_to_remove)} chunks")
        return True


def list_documents() -> List[Dict[str, Any]]:
    """Return list of all registered documents."""
    return [
        {"doc_id": doc_id, **info}
        for doc_id, info in _doc_registry.items()
    ]


def get_stats() -> Dict[str, Any]:
    """Return index statistics."""
    return {
        "total_documents": len(_doc_registry),
        "total_chunks": len(_metadata),
        "index_size": len(_vectors) if _vectors is not None else 0,
    }

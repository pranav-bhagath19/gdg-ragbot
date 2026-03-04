"""
services/vectorstore.py - Pinecone cloud vector store.

Features:
- Stores vectors in Pinecone serverless index (persisted in the cloud)
- Supports add, search, delete, list
- Survives redeployments — data lives in Pinecone, not on local disk
- Thread-safe with asyncio.Lock
"""
import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from services.embed import EMBEDDING_DIM

# Concurrency lock
_index_lock = asyncio.Lock()

# Pinecone index handle
_index = None
_initialized = False

# Local cache of document registry (also stored in Pinecone metadata)
_doc_registry: Dict[str, Dict[str, Any]] = {}


def init_store(storage_path: str = ""):
    """
    Initialize the Pinecone vector store. Call once at app startup.
    Creates the index if it doesn't exist, then connects to it.
    """
    global _index, _initialized, _doc_registry

    from pinecone import Pinecone, ServerlessSpec
    from config import get_settings

    settings = get_settings()

    if not settings.pinecone_api_key:
        logger.warning("PINECONE_API_KEY not set — vector store will not work!")
        _initialized = False
        return

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    # Create index if it doesn't exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index '{index_name}' (dim={EMBEDDING_DIM}, cosine)...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for Pinecone index to be ready...")
            time.sleep(1)

    _index = pc.Index(index_name)
    _initialized = True

    # Rebuild local doc registry from Pinecone metadata
    _rebuild_registry()

    stats = _index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)
    logger.info(f"Pinecone index '{index_name}' ready: {total_vectors} vectors, {len(_doc_registry)} documents")


def _rebuild_registry():
    """
    Rebuild the local document registry by scanning Pinecone metadata.
    Uses a list query to find all unique doc_ids and their info.
    """
    global _doc_registry

    if not _index:
        return

    _doc_registry = {}

    # Pinecone doesn't have a "list all metadata" API directly.
    # We use a dummy zero-vector query with a large top_k to fetch metadata.
    # For small datasets (< 10K chunks) this works well.
    try:
        # Query with zero vector to get all vectors (sorted by score, but we just need metadata)
        dummy_vec = [0.0] * EMBEDDING_DIM
        # Fetch up to 10000 results to rebuild registry
        results = _index.query(
            vector=dummy_vec,
            top_k=10000,
            include_metadata=True,
        )

        seen_docs = {}
        total_chunks = 0
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "filename": meta.get("filename", "unknown"),
                    "ingested_at": meta.get("ingested_at", ""),
                    "total_sections": int(meta.get("total_sections", 0)),
                    "chunk_count": 0,
                }
            if doc_id:
                seen_docs[doc_id]["chunk_count"] = seen_docs[doc_id].get("chunk_count", 0) + 1
            total_chunks += 1

        _doc_registry = seen_docs
        logger.info(f"Rebuilt registry: {len(_doc_registry)} documents, {total_chunks} chunks")
    except Exception as e:
        logger.warning(f"Could not rebuild registry from Pinecone: {e}")
        _doc_registry = {}


async def add_chunks(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    doc_id: str,
    doc_info: Dict[str, Any],
) -> int:
    """
    Add a batch of chunk embeddings + metadata to Pinecone.
    """
    async with _index_lock:
        if not _initialized or not _index:
            raise RuntimeError("Vector store not initialized. Call init_store() first.")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = embeddings / norms

        # Build Pinecone upsert batch
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            # Pinecone metadata must be flat (str, int, float, bool, list of str)
            metadata = {
                "text": chunk.get("text", "")[:40000],  # Pinecone 40KB metadata limit
                "source": str(chunk.get("source", "")),
                "page": str(chunk.get("page", "")),
                "doc_id": doc_id,
                "filename": doc_info.get("filename", ""),
                "ingested_at": doc_info.get("ingested_at", ""),
                "total_sections": int(doc_info.get("total_sections", 0)),
                "chunk_index": i,
            }
            vectors_to_upsert.append({
                "id": chunk_id,
                "values": normalized[i].tolist(),
                "metadata": metadata,
            })

        # Upsert in batches of 100 (Pinecone recommended batch size)
        batch_size = 100
        for batch_start in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[batch_start:batch_start + batch_size]
            _index.upsert(vectors=batch)

        # Update local registry
        _doc_registry[doc_id] = {
            **doc_info,
            "chunk_count": len(chunks),
        }

        logger.info(f"Upserted {len(chunks)} chunks to Pinecone for doc_id={doc_id}")
        return len(chunks)


async def search(
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search Pinecone for top-k most similar chunks.
    """
    if not _initialized or not _index:
        logger.warning("Search called but Pinecone not initialized")
        return []

    # Normalize query
    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0:
        q_norm = 1e-10
    q_vec = (query_embedding / q_norm).tolist()

    results = _index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )

    matches = []
    for match in results.get("matches", []):
        meta = dict(match.get("metadata", {}))
        meta["score"] = float(match.get("score", 0))
        meta["chunk_id"] = match.get("id", "")
        matches.append(meta)

    return matches


async def delete_document(doc_id: str) -> bool:
    """
    Delete all chunks belonging to a document from Pinecone.
    """
    async with _index_lock:
        if not _initialized or not _index:
            return False

        if doc_id not in _doc_registry:
            return False

        try:
            # Use metadata filter to find all vectors for this doc_id
            # Then delete them by ID
            # First, query to get all chunk IDs for this document
            dummy_vec = [0.0] * EMBEDDING_DIM
            results = _index.query(
                vector=dummy_vec,
                top_k=10000,
                include_metadata=True,
                filter={"doc_id": {"$eq": doc_id}},
            )

            ids_to_delete = [m["id"] for m in results.get("matches", [])]

            if ids_to_delete:
                # Delete in batches of 1000
                for i in range(0, len(ids_to_delete), 1000):
                    batch = ids_to_delete[i:i + 1000]
                    _index.delete(ids=batch)

            # Remove from local registry
            del _doc_registry[doc_id]
            logger.info(f"Deleted document {doc_id}: removed {len(ids_to_delete)} chunks from Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False


def list_documents() -> List[Dict[str, Any]]:
    """Return list of all registered documents."""
    return [
        {"doc_id": doc_id, **info}
        for doc_id, info in _doc_registry.items()
    ]


def get_stats() -> Dict[str, Any]:
    """Return index statistics."""
    if not _initialized or not _index:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "index_size": 0,
        }

    try:
        stats = _index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
    except Exception:
        total_vectors = sum(d.get("chunk_count", 0) for d in _doc_registry.values())

    return {
        "total_documents": len(_doc_registry),
        "total_chunks": total_vectors,
        "index_size": total_vectors,
    }

"""
routers/ingest.py - POST /ingest endpoint for document upload and processing.
Place this file at: rag-chatbot/routers/ingest.py

Pipeline: upload → validate → parse → chunk → embed → upsert → persist
Uses FastAPI BackgroundTasks so large files don't timeout the HTTP request.
"""
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from loguru import logger

from config import get_settings, Settings
from services.parse import parse_file
from services.chunk import RecursiveCharacterTextSplitter
from services.embed import embed_texts
from services import vectorstore

router = APIRouter()

# In-memory job status tracker {job_id: status_dict}
_jobs: Dict[str, Dict[str, Any]] = {}

ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",  # Some clients send this for any binary
}


class IngestResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    doc_id: str = ""
    chunks_added: int = 0
    message: str = ""
    created_at: str = ""
    completed_at: str = ""


async def _process_document(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    settings: Settings,
):
    """
    Background task: full ingestion pipeline.
    Updates job status at each stage.
    """
    doc_id = str(uuid.uuid4())
    _jobs[job_id]["status"] = "processing"
    _jobs[job_id]["doc_id"] = doc_id

    try:
        # Step 1: Parse
        logger.info(f"[job={job_id}] Parsing '{filename}'...")
        _jobs[job_id]["stage"] = "parsing"
        raw_docs = parse_file(file_bytes, filename)

        if not raw_docs:
            raise ValueError("No text content extracted from document")

        # Step 2: Chunk
        logger.info(f"[job={job_id}] Chunking {len(raw_docs)} sections...")
        _jobs[job_id]["stage"] = "chunking"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)

        if not chunks:
            raise ValueError("No chunks generated after splitting")

        # Step 3: Embed
        logger.info(f"[job={job_id}] Embedding {len(chunks)} chunks...")
        _jobs[job_id]["stage"] = "embedding"
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # Step 4: Upsert into vector store
        logger.info(f"[job={job_id}] Upserting into vector store...")
        _jobs[job_id]["stage"] = "storing"
        doc_info = {
            "filename": filename,
            "doc_id": doc_id,
            "ingested_at": datetime.utcnow().isoformat(),
            "total_sections": len(raw_docs),
        }
        chunks_added = await vectorstore.add_chunks(embeddings, chunks, doc_id, doc_info)

        # Done
        _jobs[job_id].update({
            "status": "completed",
            "chunks_added": chunks_added,
            "completed_at": datetime.utcnow().isoformat(),
            "message": f"Successfully ingested {chunks_added} chunks",
        })
        logger.info(f"[job={job_id}] ✅ Ingestion complete: {chunks_added} chunks for '{filename}'")

    except Exception as e:
        logger.error(f"[job={job_id}] ❌ Ingestion failed: {e}")
        _jobs[job_id].update({
            "status": "failed",
            "message": str(e),
            "completed_at": datetime.utcnow().isoformat(),
        })


@router.post("/ingest", response_model=IngestResponse, summary="Upload and ingest a document")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Upload a PDF, Excel, or DOCX file for ingestion into the RAG system.

    - File is parsed, chunked, embedded, and stored asynchronously.
    - Returns a job_id to poll for status at GET /ingest/status/{job_id}.
    - Max file size: 10MB (configurable via MAX_FILE_SIZE_MB env var).
    """
    # Validate file extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_bytes) / 1024 / 1024:.1f}MB. Max: {settings.max_file_size_mb}MB"
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Check document limit
    current_docs = vectorstore.list_documents()
    if len(current_docs) >= settings.max_docs:
        raise HTTPException(
            status_code=429,
            detail=f"Document limit reached ({settings.max_docs}). Delete some documents first."
        )

    # Create job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
        "doc_id": "",
        "chunks_added": 0,
        "message": "Queued for processing",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": "",
        "stage": "queued",
    }

    # Start background processing
    background_tasks.add_task(_process_document, job_id, file_bytes, filename, settings)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        filename=filename,
        message="Document queued for processing. Poll /ingest/status/{job_id} for updates.",
    )


@router.get("/ingest/status/{job_id}", response_model=JobStatusResponse, summary="Check ingestion job status")
async def get_job_status(job_id: str):
    """Poll this endpoint to check the status of an ingestion job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    return JobStatusResponse(**{k: v for k, v in job.items() if k in JobStatusResponse.model_fields})

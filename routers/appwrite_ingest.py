"""
routers/appwrite_ingest.py - Appwrite Storage ingestion endpoints.

Pulls documents from an Appwrite Storage bucket and feeds them
into the existing parse -> chunk -> embed -> store pipeline.
"""
import uuid
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from loguru import logger

from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.query import Query

from config import get_settings, Settings
from routers.ingest import _process_document, _jobs, ALLOWED_EXTENSIONS
from services import vectorstore

router = APIRouter()


# ─── Response Models ──────────────────────────────────────────────────────────

class AppwriteSyncResponse(BaseModel):
    message: str
    jobs: List[dict]
    skipped: List[str] = []


class AppwriteFileResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    message: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_appwrite_storage(settings: Settings) -> Storage:
    """Initialize Appwrite client and return Storage service."""
    if not all([settings.appwrite_endpoint, settings.appwrite_project_id,
                settings.appwrite_api_key, settings.appwrite_bucket_id]):
        raise HTTPException(
            status_code=400,
            detail="Appwrite is not configured. Set APPWRITE_ENDPOINT, APPWRITE_PROJECT_ID, "
                   "APPWRITE_API_KEY, and APPWRITE_BUCKET_ID in your .env file."
        )

    client = Client()
    client.set_endpoint(settings.appwrite_endpoint)
    client.set_project(settings.appwrite_project_id)
    client.set_key(settings.appwrite_api_key)

    return Storage(client)


def _create_job(filename: str) -> str:
    """Create a job entry in the shared _jobs tracker and return the job_id."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
        "doc_id": "",
        "chunks_added": 0,
        "message": "Queued for processing (from Appwrite)",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": "",
        "stage": "queued",
    }
    return job_id


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/ingest/appwrite/sync",
    response_model=AppwriteSyncResponse,
    summary="Sync all documents from Appwrite bucket",
)
async def sync_appwrite_bucket(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    List all files in the configured Appwrite Storage bucket,
    skip already-ingested documents, and queue the rest for ingestion.

    Returns job_ids that can be polled via GET /ingest/status/{job_id}.
    """
    storage = _get_appwrite_storage(settings)
    bucket_id = settings.appwrite_bucket_id

    # List ALL files in bucket (handle pagination — Appwrite defaults to 25)
    try:
        all_files = []
        offset = 0
        limit = 100  # Max per request
        while True:
            result = storage.list_files(bucket_id=bucket_id, queries=[
                Query.limit(limit),
                Query.offset(offset),
            ])
            batch = result.get("files", [])
            all_files.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to list Appwrite files: {e}")

    files = all_files
    if not files:
        return AppwriteSyncResponse(message="No files found in Appwrite bucket.", jobs=[])

    logger.info(f"Found {len(files)} files in Appwrite bucket")

    # Get already-ingested filenames to skip duplicates
    existing_docs = vectorstore.list_documents()
    existing_filenames = {doc.get("filename", "") for doc in existing_docs}
    logger.info(f"Already ingested: {existing_filenames}")

    # Check document limit
    current_count = len(existing_docs)
    max_docs = settings.max_docs

    jobs = []
    skipped = []

    for file_info in files:
        filename = file_info.get("name", "")
        file_id = file_info.get("$id", "")
        ext = Path(filename).suffix.lower()

        logger.info(f"Processing: '{filename}' (id={file_id}, ext={ext})")

        # Skip unsupported extensions
        if ext not in ALLOWED_EXTENSIONS:
            skipped.append(f"{filename} (unsupported type: {ext})")
            continue

        # Skip already-ingested files
        if filename in existing_filenames:
            skipped.append(f"{filename} (already ingested)")
            continue

        # Check document limit
        if current_count >= max_docs:
            skipped.append(f"{filename} (document limit reached: {max_docs})")
            continue

        # Download file bytes from Appwrite
        try:
            file_bytes = storage.get_file_download(bucket_id=bucket_id, file_id=file_id)
        except Exception as e:
            logger.error(f"Failed to download '{filename}' from Appwrite: {e}")
            skipped.append(f"{filename} (download failed: {e})")
            continue

        # Validate size
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            skipped.append(f"{filename} (too large: {len(file_bytes) / 1024 / 1024:.1f}MB)")
            continue

        if len(file_bytes) == 0:
            skipped.append(f"{filename} (empty file)")
            continue

        # Queue for ingestion
        job_id = _create_job(filename)
        background_tasks.add_task(_process_document, job_id, file_bytes, filename, settings)
        jobs.append({"job_id": job_id, "filename": filename})
        current_count += 1

    return AppwriteSyncResponse(
        message=f"Queued {len(jobs)} file(s) for ingestion.",
        jobs=jobs,
        skipped=skipped,
    )


@router.post(
    "/ingest/appwrite/{file_id}",
    response_model=AppwriteFileResponse,
    summary="Ingest a single file from Appwrite by file ID",
)
async def ingest_appwrite_file(
    file_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Download a specific file from Appwrite Storage and ingest it into the RAG system.

    - Get the file_id from your Appwrite Console or via /ingest/appwrite/sync.
    - Returns a job_id to poll for status at GET /ingest/status/{job_id}.
    """
    storage = _get_appwrite_storage(settings)
    bucket_id = settings.appwrite_bucket_id

    # Get file metadata
    try:
        file_meta = storage.get_file(bucket_id=bucket_id, file_id=file_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found in Appwrite: {e}")

    filename = file_meta.get("name", "unknown")
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check document limit
    current_docs = vectorstore.list_documents()
    if len(current_docs) >= settings.max_docs:
        raise HTTPException(
            status_code=429,
            detail=f"Document limit reached ({settings.max_docs}). Delete some documents first."
        )

    # Download file bytes
    try:
        file_bytes = storage.get_file_download(bucket_id=bucket_id, file_id=file_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download file from Appwrite: {e}")

    # Validate size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_bytes) / 1024 / 1024:.1f}MB. Max: {settings.max_file_size_mb}MB"
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file in Appwrite")

    # Queue for ingestion
    job_id = _create_job(filename)
    background_tasks.add_task(_process_document, job_id, file_bytes, filename, settings)

    return AppwriteFileResponse(
        job_id=job_id,
        status="queued",
        filename=filename,
        message=f"File '{filename}' queued for processing. Poll /ingest/status/{job_id} for updates.",
    )


@router.post(
    "/ingest/appwrite/webhook",
    summary="Appwrite webhook receiver for auto-ingestion",
)
async def appwrite_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Webhook endpoint for Appwrite to call when a new file is uploaded.

    Setup in Appwrite Console:
    1. Go to your project > Settings > Webhooks
    2. Create webhook with URL: https://<your-domain>/ingest/appwrite/webhook
    3. Select event: storage.files.create
    """
    storage = _get_appwrite_storage(settings)
    bucket_id = settings.appwrite_bucket_id

    # Parse the webhook payload
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Appwrite webhook payload contains the file object directly
    file_id = payload.get("$id", "")
    filename = payload.get("name", "")

    if not file_id:
        raise HTTPException(status_code=400, detail="Missing file ID in webhook payload")

    logger.info(f"[webhook] Received Appwrite event for file: '{filename}' (id={file_id})")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.info(f"[webhook] Skipping unsupported file type: {ext}")
        return {"status": "skipped", "reason": f"Unsupported file type: {ext}"}

    # Check document limit
    current_docs = vectorstore.list_documents()
    if len(current_docs) >= settings.max_docs:
        logger.warning(f"[webhook] Document limit reached ({settings.max_docs})")
        return {"status": "skipped", "reason": "Document limit reached"}

    # Download and ingest
    try:
        file_bytes = storage.get_file_download(bucket_id=bucket_id, file_id=file_id)
    except Exception as e:
        logger.error(f"[webhook] Failed to download '{filename}': {e}")
        raise HTTPException(status_code=502, detail=f"Failed to download file: {e}")

    if len(file_bytes) == 0:
        return {"status": "skipped", "reason": "Empty file"}

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return {"status": "skipped", "reason": f"File too large: {len(file_bytes) / 1024 / 1024:.1f}MB"}

    job_id = _create_job(filename)
    background_tasks.add_task(_process_document, job_id, file_bytes, filename, settings)

    logger.info(f"[webhook] Queued '{filename}' for ingestion (job_id={job_id})")
    return {"status": "queued", "job_id": job_id, "filename": filename}

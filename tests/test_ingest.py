"""
tests/test_ingest.py - Integration tests for the /ingest endpoint.
Place this file at: rag-chatbot/tests/test_ingest.py
"""
import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_health_check(client):
    """Health check should return 200 with correct structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data
    assert "providers" in data


def test_ingest_docx(client, sample_docx_bytes):
    """Test DOCX file ingestion returns a job_id."""
    with patch("services.embed.embed_texts") as mock_embed:
        mock_embed.return_value = np.random.rand(5, 384).astype("float32")

        response = client.post(
            "/ingest",
            files={"file": ("test.docx", sample_docx_bytes,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["filename"] == "test.docx"
    assert data["status"] in ("queued", "processing", "completed")


def test_ingest_xlsx(client, sample_xlsx_bytes):
    """Test XLSX file ingestion returns a job_id."""
    with patch("services.embed.embed_texts") as mock_embed:
        mock_embed.return_value = np.random.rand(3, 384).astype("float32")

        response = client.post(
            "/ingest",
            files={"file": ("test.xlsx", sample_xlsx_bytes,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["filename"] == "test.xlsx"


def test_ingest_unsupported_type(client):
    """Test that unsupported file types are rejected."""
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"some text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_ingest_empty_file(client):
    """Test that empty files are rejected."""
    response = client.post(
        "/ingest",
        files={"file": ("test.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_ingest_file_too_large(client):
    """Test that oversized files are rejected."""
    # Create a 11MB file
    big_content = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/ingest",
        files={"file": ("big.pdf", big_content, "application/pdf")},
    )
    assert response.status_code == 413


def test_job_status_not_found(client):
    """Test that querying a non-existent job returns 404."""
    response = client.get("/ingest/status/nonexistent-job-id")
    assert response.status_code == 404


def test_job_status_tracking(client, sample_docx_bytes):
    """Test that job status can be polled after ingestion."""
    with patch("services.embed.embed_texts") as mock_embed:
        mock_embed.return_value = np.random.rand(3, 384).astype("float32")

        # Ingest a document
        response = client.post(
            "/ingest",
            files={"file": ("track_test.docx", sample_docx_bytes,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll status (may need brief delay for background task)
        time.sleep(0.5)
        status_response = client.get(f"/ingest/status/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ("queued", "processing", "completed", "failed")

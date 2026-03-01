"""
tests/test_documents.py - Tests for GET /documents and DELETE /documents/{id}.
Place this file at: rag-chatbot/tests/test_documents.py
"""
import pytest
from unittest.mock import patch


def test_list_documents_empty(client):
    """List documents on empty store returns empty list."""
    with patch("services.vectorstore.list_documents") as mock_list:
        mock_list.return_value = []
        response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == []
    assert data["total"] == 0


def test_list_documents_with_data(client):
    """List documents with sample data."""
    mock_docs = [
        {
            "doc_id": "abc-123",
            "filename": "report.pdf",
            "ingested_at": "2024-01-01T00:00:00",
            "chunk_count": 15,
            "total_sections": 5,
        },
        {
            "doc_id": "def-456",
            "filename": "data.xlsx",
            "ingested_at": "2024-01-02T00:00:00",
            "chunk_count": 8,
            "total_sections": 3,
        }
    ]

    with patch("services.vectorstore.list_documents") as mock_list:
        mock_list.return_value = mock_docs
        response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["documents"]) == 2
    assert data["documents"][0]["filename"] == "report.pdf"


def test_delete_document_success(client):
    """Successful deletion returns 200."""
    with patch("services.vectorstore.delete_document") as mock_delete:
        mock_delete.return_value = True
        response = client.delete("/documents/abc-123")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["doc_id"] == "abc-123"


def test_delete_document_not_found(client):
    """Deleting non-existent document returns 404."""
    with patch("services.vectorstore.delete_document") as mock_delete:
        mock_delete.return_value = False
        response = client.delete("/documents/nonexistent-id")

    assert response.status_code == 404


def test_documents_response_structure(client):
    """Verify response structure matches expected schema."""
    mock_docs = [{
        "doc_id": "xyz-789",
        "filename": "manual.docx",
        "ingested_at": "2024-06-01T12:00:00",
        "chunk_count": 20,
        "total_sections": 8,
    }]

    with patch("services.vectorstore.list_documents", return_value=mock_docs):
        response = client.get("/documents")

    assert response.status_code == 200
    doc = response.json()["documents"][0]
    assert "doc_id" in doc
    assert "filename" in doc
    assert "ingested_at" in doc
    assert "chunk_count" in doc

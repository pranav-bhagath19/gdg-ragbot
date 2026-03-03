"""
tests/test_chat.py - Integration tests for the /chat endpoint.
Place this file at: rag-chatbot/tests/test_chat.py
"""
import numpy as np
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def test_chat_no_documents(client):
    """Chat with no documents should return 422."""
    # Ensure no chunks are in the store
    with patch("services.vectorstore.get_stats") as mock_stats:
        mock_stats.return_value = {"total_documents": 0, "total_chunks": 0, "index_size": 0}

        response = client.post("/chat", json={"query": "What is this about?"})
        assert response.status_code == 422


def test_chat_with_mock_data(client):
    """Test chat returns valid response structure when documents exist."""
    mock_chunks = [
        {
            "text": "The sky is blue because of Rayleigh scattering.",
            "source": "science.pdf",
            "page": 1,
            "chunk_index": 0,
            "score": 0.92,
            "doc_id": "test-doc",
            "chunk_id": "test-chunk",
            "type": "pdf",
        }
    ]

    with patch("services.vectorstore.get_stats") as mock_stats, \
         patch("routers.chat.retrieve") as mock_retrieve, \
         patch("routers.chat.call_llm_with_failover") as mock_llm:

        mock_stats.return_value = {"total_documents": 1, "total_chunks": 5, "index_size": 5}
        mock_retrieve.return_value = mock_chunks
        mock_llm.return_value = (
            "The sky is blue due to Rayleigh scattering [Source 1].",
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "fallback_used": False,
                "latency_ms": 300,
                "prompt_tokens": 100,
                "completion_tokens": 30,
            }
        )

        response = client.post("/chat", json={"query": "Why is the sky blue?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "provider" in data
    assert "fallback_used" in data
    assert "latency_ms" in data
    assert data["provider"] == "groq"
    assert data["fallback_used"] is False
    assert len(data["citations"]) > 0


def test_chat_with_history(client):
    """Test that conversation history is accepted."""
    mock_chunks = [{"text": "Test content", "source": "doc.pdf", "page": 1,
                    "chunk_index": 0, "score": 0.9, "doc_id": "d1", "chunk_id": "c1", "type": "pdf"}]

    with patch("services.vectorstore.get_stats") as mock_stats, \
         patch("routers.chat.retrieve", return_value=mock_chunks), \
         patch("routers.chat.call_llm_with_failover") as mock_llm:

        mock_stats.return_value = {"total_documents": 1, "total_chunks": 5, "index_size": 5}
        mock_llm.return_value = ("Answer with history context.", {
            "provider": "gemini", "model": "gemini-1.5-flash",
            "fallback_used": True, "latency_ms": 500,
            "prompt_tokens": 120, "completion_tokens": 25,
        })

        response = client.post("/chat", json={
            "query": "Tell me more",
            "history": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "This is a test document."},
            ]
        })

    assert response.status_code == 200
    data = response.json()
    assert data["fallback_used"] is True
    assert data["provider"] == "gemini"


def test_chat_invalid_query(client):
    """Empty query should be rejected."""
    response = client.post("/chat", json={"query": ""})
    assert response.status_code == 422


def test_chat_query_too_long(client):
    """Query over 2000 chars should be rejected."""
    response = client.post("/chat", json={"query": "a" * 2001})
    assert response.status_code == 422


def test_chat_llm_failure_returns_503(client):
    """If all LLM providers fail, return 503."""
    mock_chunks = [{"text": "content", "source": "doc.pdf", "page": 1,
                    "chunk_index": 0, "score": 0.9, "doc_id": "d1", "chunk_id": "c1", "type": "pdf"}]

    with patch("services.vectorstore.get_stats") as mock_stats, \
         patch("routers.chat.retrieve", return_value=mock_chunks), \
         patch("routers.chat.call_llm_with_failover") as mock_llm:

        mock_stats.return_value = {"total_documents": 1, "total_chunks": 5, "index_size": 5}
        mock_llm.side_effect = RuntimeError("All providers failed")

        response = client.post("/chat", json={"query": "Test question?"})

    assert response.status_code == 503

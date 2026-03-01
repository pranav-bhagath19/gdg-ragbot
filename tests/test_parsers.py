"""
tests/test_parsers.py - Unit tests for file parsers (parse.py).
Place this file at: rag-chatbot/tests/test_parsers.py
"""
import io
import pytest


def test_parse_docx(sample_docx_bytes):
    """DOCX parser should return list of dicts with text."""
    from services.parse import parse_docx

    results = parse_docx(sample_docx_bytes, "test.docx")
    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "text" in item
        assert "source" in item
        assert "page" in item
        assert item["source"] == "test.docx"
        assert item["type"] == "docx"
        assert len(item["text"]) > 0


def test_parse_xlsx(sample_xlsx_bytes):
    """XLSX parser should convert rows to natural language strings."""
    from services.parse import parse_xlsx

    results = parse_xlsx(sample_xlsx_bytes, "test.xlsx")
    assert isinstance(results, list)
    assert len(results) > 0

    # Content should include column names and values
    all_text = " ".join([r["text"] for r in results])
    assert "Name" in all_text or "Alice" in all_text


def test_parse_unsupported_extension():
    """Unsupported extension should raise ValueError."""
    from services.parse import parse_file

    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file(b"some content", "file.txt")


def test_chunk_splitting():
    """Chunker should split long text into chunks with overlap."""
    from services.chunk import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    long_text = "word " * 200  # ~1000 chars

    docs = [{"text": long_text, "page": 1, "source": "test.txt", "type": "txt"}]
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert chunk["source"] == "test.txt"
        assert len(chunk["text"]) > 0


def test_chunk_short_text_no_split():
    """Short text should not be split."""
    from services.chunk import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = [{"text": "Short text", "page": 1, "source": "s.txt", "type": "txt"}]
    chunks = splitter.split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text"


def test_retriever_format_context():
    """format_context should produce citation list and context string."""
    from services.retriever import format_context

    chunks = [
        {"text": "Python is a programming language.", "source": "python.pdf",
         "page": 1, "chunk_index": 0, "score": 0.95},
        {"text": "It was created by Guido van Rossum.", "source": "python.pdf",
         "page": 2, "chunk_index": 0, "score": 0.88},
    ]

    context, citations = format_context(chunks)

    assert "Python is a programming language" in context
    assert "Source 1" in context
    assert len(citations) == 2
    assert citations[0]["document"] == "python.pdf"
    assert citations[0]["relevance_score"] == 0.95


def test_prompt_builder():
    """build_prompt should return properly structured messages."""
    from services.prompt import build_prompt

    messages = build_prompt(
        query="What is Python?",
        context="[Source 1] Python is a language.",
        history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )

    assert isinstance(messages, list)
    assert len(messages) >= 1
    last_msg = messages[-1]
    assert last_msg["role"] == "user"
    assert "What is Python?" in last_msg["content"]
    assert "Python is a language" in last_msg["content"]

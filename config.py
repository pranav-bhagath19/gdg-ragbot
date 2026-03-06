"""
config.py - Environment configuration.
Place this file at: rag-chatbot/config.py
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Groq Keys (primary + backup) ──────────────────────────
    groq_api_key: str = ""
    groq_api_key_backup: str = ""

    # ── Gemini Keys (primary + backup) ────────────────────────
    gemini_api_key: str = ""
    gemini_api_key_backup: str = ""

    # ── Storage ───────────────────────────────────────────────
    storage_path: str = "./data"
    pdf_folder: str = ""

    # ── Limits ────────────────────────────────────────────────
    max_file_size_mb: int = 10
    max_docs: int = 100

    # ── App ───────────────────────────────────────────────────
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── LLM settings ──────────────────────────────────────────
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-1.5-flash"
    max_tokens: int = 1024
    top_k_chunks: int = 15

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Rate limiting ─────────────────────────────────────────
    rate_limit_per_minute: int = 30

    # ── Pinecone ──────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "ragbot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
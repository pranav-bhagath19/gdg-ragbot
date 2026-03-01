"""
config.py - Environment configuration using pydantic-settings.
Place this file at: rag-chatbot/config.py
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # LLM API Keys
    groq_api_key: str = ""
    gemini_api_key: str = ""

    # Storage
    storage_path: str = "./data"
    pdf_folder: str = "./data"

    # Limits
    max_file_size_mb: int = 10
    max_docs: int = 100

    # App settings
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = False

    # LLM settings
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-1.5-flash"
    max_tokens: int = 1024
    top_k_chunks: int = 5

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Rate limiting
    rate_limit_per_minute: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()

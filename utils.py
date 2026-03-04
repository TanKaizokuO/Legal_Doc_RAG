"""
utils.py — Shared utilities for the AI Lawyer RAG application.
Handles configuration loading, logging, and display helpers.
"""

import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# ── Load .env on import ────────────────────────────────────────────────────────
load_dotenv()


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s — %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── Config accessors ──────────────────────────────────────────────────────────
def get_llm_provider() -> str:
    """Return the LLM provider: 'openai' or 'ollama'."""
    return os.getenv("LLM_PROVIDER", "openai").lower()


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-..."):
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example → .env and add your key, "
            "or set LLM_PROVIDER=ollama to use a local model."
        )
    return key


def get_chunk_config() -> dict:
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
    }


def get_top_k() -> int:
    return int(os.getenv("TOP_K", 4))


def get_chroma_path() -> str:
    return os.getenv("CHROMA_DB_PATH", "./chroma_db")


def get_openai_chat_model() -> str:
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")


def get_openai_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")


def get_ollama_config() -> dict:
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_CHAT_MODEL", "llama3"),
    }


# ── Display helpers ───────────────────────────────────────────────────────────
def format_sources(source_docs) -> List[dict]:
    """
    Convert a list of LangChain Documents into a list of dicts
    suitable for display in the Streamlit UI.
    """
    seen_content: set = set()
    formatted = []
    for doc in source_docs:
        content = doc.page_content.strip()
        # Deduplicate identical chunks (can happen with overlap)
        if content in seen_content:
            continue
        seen_content.add(content)
        meta = doc.metadata or {}
        formatted.append(
            {
                "content": content,
                "page": meta.get("page", "N/A"),
                "source": Path(meta.get("source", "document")).name,
            }
        )
    return formatted


def ensure_data_directory() -> Path:
    """Make sure the ./data directory exists and return its Path."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

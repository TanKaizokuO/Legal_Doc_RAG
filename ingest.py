"""
ingest.py — Document Ingestion Pipeline for the AI Lawyer RAG Application.

Responsibilities:
  1. Load PDF documents using PyPDF.
  2. Split text into overlapping chunks.
  3. Generate embeddings (OpenAI or HuggingFace).
  4. Persist chunks + embeddings in ChromaDB.

Usage (CLI):
    python ingest.py --file path/to/document.pdf [--collection my_collection]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from utils import (
    get_chunk_config,
    get_chroma_path,
    get_llm_provider,
    get_logger,
    get_openai_embedding_model,
)

log = get_logger(__name__)

# ── Embedding factory ──────────────────────────────────────────────────────────

def get_embeddings():
    """
    Return the appropriate embedding model based on LLM_PROVIDER.
      - openai  → OpenAIEmbeddings (text-embedding-ada-002)
      - ollama  → HuggingFaceEmbeddings (all-MiniLM-L6-v2, free & local)
    """
    provider = get_llm_provider()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        from utils import get_openai_api_key
        log.info("Using OpenAI embeddings — model: %s", get_openai_embedding_model())
        return OpenAIEmbeddings(
            model=get_openai_embedding_model(),
            openai_api_key=get_openai_api_key(),
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        log.info("Using HuggingFace embeddings — model: %s", model_name)
        return HuggingFaceEmbeddings(model_name=model_name)


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return a list of LangChain Documents (one per page).
    Each Document carries metadata: {source, page}.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    log.info("Loading PDF: %s", path.name)
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    log.info("Loaded %d page(s) from '%s'", len(documents), path.name)
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks using
    RecursiveCharacterTextSplitter — preserves sentence boundaries.
    """
    cfg = get_chunk_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    log.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(chunks),
        cfg["chunk_size"],
        cfg["chunk_overlap"],
    )
    return chunks


def create_vectorstore(
    chunks: List[Document],
    collection_name: str = "legal_docs",
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Embed chunks and store them in a persistent ChromaDB collection.
    Returns the Chroma vectorstore instance.
    """
    persist_dir = persist_directory or get_chroma_path()
    embeddings = get_embeddings()

    log.info(
        "Creating ChromaDB vectorstore — collection='%s', path='%s'",
        collection_name,
        persist_dir,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    log.info(
        "Stored %d chunks in ChromaDB at '%s'", len(chunks), persist_dir
    )
    return vectorstore


def load_vectorstore(
    collection_name: str = "legal_docs",
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Load an existing ChromaDB collection from disk.
    Raises RuntimeError if the collection doesn't exist yet.
    """
    persist_dir = persist_directory or get_chroma_path()
    if not Path(persist_dir).exists():
        raise RuntimeError(
            f"No ChromaDB found at '{persist_dir}'. "
            "Please ingest a document first."
        )
    embeddings = get_embeddings()
    log.info(
        "Loading ChromaDB — collection='%s', path='%s'",
        collection_name,
        persist_dir,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def ingest_pdf(
    file_path: str,
    collection_name: str = "legal_docs",
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    High-level convenience function: load → split → embed → store.
    Returns the ready-to-use Chroma vectorstore.
    """
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(
        chunks,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return vectorstore


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Lawyer — Ingest a PDF into ChromaDB"
    )
    parser.add_argument("--file", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--collection",
        default="legal_docs",
        help="ChromaDB collection name (default: legal_docs)",
    )
    args = parser.parse_args()

    try:
        vs = ingest_pdf(args.file, collection_name=args.collection)
        print(f"\n✅  Document ingested successfully into collection '{args.collection}'.")
        print(f"    ChromaDB path: {get_chroma_path()}")
        count = vs._collection.count()
        print(f"    Total chunks stored: {count}")
    except Exception as exc:
        log.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

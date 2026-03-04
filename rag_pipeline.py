"""
rag_pipeline.py — Retrieval-Augmented Generation (RAG) Pipeline.

Responsibilities:
  1. Build a retriever from the ChromaDB vectorstore.
  2. Configure the LLM (OpenAI or Ollama).
  3. Construct a RetrievalQA chain with a legal-focused system prompt.
  4. Expose a simple answer_question() interface.
"""

from typing import Any, Dict, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from utils import (
    get_llm_provider,
    get_logger,
    get_ollama_config,
    get_openai_chat_model,
    get_top_k,
    format_sources,
)

log = get_logger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

LEGAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a highly skilled AI legal assistant specialized in analyzing \
contracts, agreements, NDAs, policies, and other legal documents.

Use ONLY the following excerpts from the legal document to answer the question. \
Do NOT use any prior knowledge outside of the provided context.

If the document does not contain enough information to answer the question, clearly \
state: "The provided document does not contain sufficient information to answer this \
question."

When answering:
- Be precise and cite specific clauses or sections when possible.
- If summarizing, be thorough and organized.
- Use plain English while preserving important legal terminology.
- Do not make legal recommendations or provide legal advice.

────────────────────────────────────────
DOCUMENT EXCERPTS:
{context}
────────────────────────────────────────

QUESTION: {question}

ANSWER:""",
)


# ── LLM factory ───────────────────────────────────────────────────────────────

def get_llm():
    """
    Return the configured LLM based on LLM_PROVIDER env var.
      - openai → ChatOpenAI (gpt-3.5-turbo or gpt-4o)
      - ollama → Ollama (llama3, mistral, etc. — must be running locally)
    """
    provider = get_llm_provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        from utils import get_openai_api_key
        model = get_openai_chat_model()
        log.info("Using OpenAI LLM — model: %s", model)
        return ChatOpenAI(
            model_name=model,
            temperature=0,           # deterministic for legal Q&A
            openai_api_key=get_openai_api_key(),
        )
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        cfg = get_ollama_config()
        log.info("Using Ollama LLM — model: %s @ %s", cfg["model"], cfg["base_url"])
        return Ollama(base_url=cfg["base_url"], model=cfg["model"], temperature=0)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. Choose 'openai' or 'ollama'."
        )


# ── Retriever factory ─────────────────────────────────────────────────────────

def build_retriever(vectorstore: Chroma):
    """
    Build a Maximum Marginal Relevance (MMR) retriever for diverse results.
    MMR balances relevance with diversity, reducing repetitive chunks.
    """
    top_k = get_top_k()
    log.info("Building MMR retriever — top_k=%d", top_k)
    return vectorstore.as_retriever(
        search_type="mmr",                  # diversity-aware retrieval
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )


# ── QA Chain factory ──────────────────────────────────────────────────────────

def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """
    Assemble the full RetrievalQA chain:
      vectorstore → retriever → prompt → LLM → structured answer
    """
    llm = get_llm()
    retriever = build_retriever(vectorstore)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                 # stuff all retrieved chunks into one prompt
        retriever=retriever,
        return_source_documents=True,       # needed to display source passages in UI
        chain_type_kwargs={"prompt": LEGAL_QA_PROMPT},
    )
    log.info("RetrievalQA chain ready.")
    return chain


# ── High-level interface ───────────────────────────────────────────────────────

def answer_question(chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Run the RAG chain on a user question.

    Returns:
        {
            "answer": str,                   # LLM-generated answer
            "sources": [                     # retrieved document chunks
                {"content": str, "page": int|str, "source": str},
                ...
            ]
        }
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    log.info("Processing question: %s", question[:120])
    result = chain.invoke({"query": question})
    answer = result.get("result", "").strip()
    raw_sources = result.get("source_documents", [])
    sources = format_sources(raw_sources)

    log.info("Answer generated (%d chars), %d source chunks", len(answer), len(sources))
    return {"answer": answer, "sources": sources}

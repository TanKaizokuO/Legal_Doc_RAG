"""
app.py — AI Lawyer: Legal Document Q&A — Streamlit Application.

Entry point:
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from ingest import ingest_pdf, load_vectorstore
from rag_pipeline import answer_question, build_qa_chain
from utils import ensure_data_directory, get_chroma_path

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Lawyer — Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, rgba(99,102,241,0.25) 0%, rgba(168,85,247,0.25) 100%);
        border: 1px solid rgba(168,85,247,0.3);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #e2e8f0; margin: 0; }
    .hero p  { font-size: 1.05rem; color: #94a3b8; margin-top: 8px; }

    /* ── Answer box ── */
    .answer-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.12) 100%);
        border: 1px solid rgba(168,85,247,0.35);
        border-radius: 14px;
        padding: 24px;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.75;
        white-space: pre-wrap;
    }

    /* ── Source chip ── */
    .source-chip {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #a5b4fc;
        margin-right: 6px;
    }

    /* ── Success / warning banners ── */
    .banner-success {
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
        border-radius: 10px;
        padding: 12px 18px;
        color: #6ee7b7;
        font-size: 0.9rem;
    }
    .banner-warn {
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.35);
        border-radius: 10px;
        padding: 12px 18px;
        color: #fcd34d;
        font-size: 0.9rem;
    }

    /* ── Input overrides ── */
    textarea, input[type="text"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ── Expander header ── */
    .st-expander > summary {
        font-size: 0.92rem;
        color: #a5b4fc;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ AI Lawyer")
    st.markdown("---")
    st.markdown("### 📂 Upload Document")

    uploaded_file = st.file_uploader(
        label="Upload a PDF legal document",
        type=["pdf"],
        help="Supports contracts, NDAs, agreements, policies, and more.",
        label_visibility="collapsed",
    )

    process_btn = st.button("⚡ Process Document", use_container_width=True, type="primary")

    # ── Process uploaded PDF ────────────────────────────────────────────────
    if process_btn:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Parsing, chunking & embedding…"):
                try:
                    ensure_data_directory()
                    # Write the uploaded bytes to a temp file for PyPDFLoader
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False, dir="data"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    vs = ingest_pdf(tmp_path, collection_name="legal_docs")
                    chain = build_qa_chain(vs)

                    st.session_state.vectorstore = vs
                    st.session_state.qa_chain = chain
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.chat_history = []   # reset on new doc

                    Path(tmp_path).unlink(missing_ok=True)  # clean up temp file
                    st.success(f"✅ **{uploaded_file.name}** processed!")
                except Exception as exc:
                    st.error(f"❌ Ingestion failed:\n\n{exc}")

    # ── Load existing DB ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗄️ Or Load Existing DB")
    load_btn = st.button("Load ChromaDB from disk", use_container_width=True)
    if load_btn:
        try:
            with st.spinner("Loading vectorstore…"):
                vs = load_vectorstore(collection_name="legal_docs")
                chain = build_qa_chain(vs)
                st.session_state.vectorstore = vs
                st.session_state.qa_chain = chain
                st.session_state.doc_name = "Previously ingested document"
            st.success("✅ Vectorstore loaded!")
        except Exception as exc:
            st.error(f"❌ {exc}")

    # ── Status ──────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.session_state.doc_name:
        st.markdown(
            f'<div class="banner-success">📄 Active: <strong>{st.session_state.doc_name}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-warn">⚠️ No document loaded. Upload a PDF to begin.</div>',
            unsafe_allow_html=True,
        )

    # ── Example queries ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "What are the termination clauses?",
        "What are the payment terms?",
        "Who is responsible for liabilities?",
        "Summarize this agreement.",
        "What are the confidentiality obligations?",
        "What happens in case of a dispute?",
        "What are the governing law and jurisdiction?",
        "Are there any indemnification provisions?",
    ]
    for q in example_queries:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state["prefill_question"] = q

    # ── Clear history ───────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main Area ──────────────────────────────────────────────────────────────────

# Hero banner
st.markdown(
    """
    <div class="hero">
        <h1>⚖️ AI Lawyer</h1>
        <p>Upload a legal document and ask anything — contracts, NDAs, agreements, policies &amp; more.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Chat history ───────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("### 📜 Conversation")
    for entry in st.session_state.chat_history:
        # User bubble
        with st.chat_message("user"):
            st.markdown(entry["question"])
        # Assistant bubble
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(
                f'<div class="answer-box">{entry["answer"]}</div>',
                unsafe_allow_html=True,
            )
            # Source passages
            if entry.get("sources"):
                with st.expander(f"📄 View {len(entry['sources'])} source passage(s)", expanded=False):
                    for i, src in enumerate(entry["sources"], 1):
                        page_label = src["page"]
                        if isinstance(page_label, int):
                            page_label += 1  # 0-indexed → human-friendly
                        st.markdown(
                            f'<span class="source-chip">📑 {src["source"]}</span>'
                            f'<span class="source-chip">Page {page_label}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"```\n{src['content']}\n```")
                        if i < len(entry["sources"]):
                            st.divider()

# ── Question input ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Question")

# Handle prefill from example buttons
default_question = st.session_state.pop("prefill_question", "")

question = st.text_area(
    label="Your legal question",
    value=default_question,
    placeholder="e.g. What are the termination clauses in this contract?",
    height=100,
    label_visibility="collapsed",
    key="question_input",
)

col1, col2 = st.columns([1, 6])
with col1:
    ask_btn = st.button("Ask ⚡", type="primary", use_container_width=True)

# ── Run the RAG pipeline ───────────────────────────────────────────────────────
if ask_btn and question.strip():
    if st.session_state.qa_chain is None:
        st.warning("⚠️ No document loaded. Please upload and process a PDF first.")
    else:
        with st.spinner("🔎 Searching document and generating answer…"):
            try:
                result = answer_question(st.session_state.qa_chain, question)
                # Append to history
                st.session_state.chat_history.append(
                    {
                        "question": question.strip(),
                        "answer": result["answer"],
                        "sources": result["sources"],
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"❌ An error occurred:\n\n{exc}")

elif ask_btn and not question.strip():
    st.warning("Please enter a question before clicking Ask.")

# ── Empty state ─────────────────────────────────────────────────────────────────
if not st.session_state.chat_history and st.session_state.doc_name is None:
    st.markdown("---")
    st.markdown(
        """
        <div class="card" style="text-align:center; color:#94a3b8;">
            <h3 style="color:#e2e8f0;">🚀 Getting Started</h3>
            <ol style="text-align:left; display:inline-block;">
                <li>Upload a PDF in the sidebar</li>
                <li>Click <strong>Process Document</strong></li>
                <li>Type your question below or pick an example query</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.82rem;'>"
    "⚖️ AI Lawyer — For informational purposes only. "
    "This is not legal advice. Always consult a qualified attorney."
    "</p>",
    unsafe_allow_html=True,
)

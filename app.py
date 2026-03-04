"""
app.py — AI Lawyer: Legal Document Q&A — Streamlit Application.
Improved UI: chat interface, status panel, theme toggle, JSON export,
document summary, source viewer, and robust error handling.

Entry point:
    streamlit run app.py
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from ingest import ingest_pdf, load_vectorstore
from rag_pipeline import answer_question, build_qa_chain
from utils import ensure_data_directory, get_chroma_path, get_llm_provider

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Lawyer — Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/TanKaizokuO/Legal_Doc_RAG",
        "Report a bug": "https://github.com/TanKaizokuO/Legal_Doc_RAG/issues",
        "About": "⚖️ AI Lawyer — RAG-powered legal document assistant.",
    },
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "vectorstore": None,
    "qa_chain": None,
    "doc_name": None,
    "doc_chunks": 0,          # total chunks stored in ChromaDB
    "chat_history": [],       # list of {question, answer, sources, timestamp}
    "dark_mode": True,        # theme toggle
    "prefill_question": "",
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════════════
#  THEME — CSS injected based on dark_mode flag
# ═══════════════════════════════════════════════════════════════════════════════
def inject_css(dark: bool) -> None:
    if dark:
        bg        = "linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%)"
        sidebar   = "rgba(255,255,255,0.04)"
        border_sb = "rgba(255,255,255,0.08)"
        card_bg   = "rgba(255,255,255,0.06)"
        card_bdr  = "rgba(255,255,255,0.10)"
        text_main = "#e2e8f0"
        text_mute = "#94a3b8"
        text_foot = "#475569"
        hero_bg   = "linear-gradient(135deg,rgba(99,102,241,0.25),rgba(168,85,247,0.25))"
        hero_bdr  = "rgba(168,85,247,0.30)"
        ans_bg    = "linear-gradient(135deg,rgba(99,102,241,0.12),rgba(168,85,247,0.12))"
        ans_bdr   = "rgba(168,85,247,0.35)"
        chip_bg   = "rgba(99,102,241,0.20)"
        chip_bdr  = "rgba(99,102,241,0.40)"
        chip_txt  = "#a5b4fc"
        stat_bg   = "rgba(255,255,255,0.05)"
        stat_bdr  = "rgba(255,255,255,0.10)"
        inp_bg    = "rgba(255,255,255,0.05)"
        inp_bdr   = "rgba(255,255,255,0.12)"
        inp_color = "#e2e8f0"
        scrl_thm  = "rgba(99,102,241,0.4)"
        warn_bg   = "rgba(245,158,11,0.12)"
        warn_bdr  = "rgba(245,158,11,0.35)"
        warn_txt  = "#fcd34d"
        ok_bg     = "rgba(16,185,129,0.12)"
        ok_bdr    = "rgba(16,185,129,0.35)"
        ok_txt    = "#6ee7b7"
    else:
        bg        = "linear-gradient(135deg, #f0f4ff 0%, #e8eeff 50%, #f5f0ff 100%)"
        sidebar   = "rgba(99,102,241,0.06)"
        border_sb = "rgba(99,102,241,0.15)"
        card_bg   = "rgba(255,255,255,0.85)"
        card_bdr  = "rgba(99,102,241,0.18)"
        text_main = "#1e1b4b"
        text_mute = "#475569"
        text_foot = "#64748b"
        hero_bg   = "linear-gradient(135deg,rgba(99,102,241,0.12),rgba(168,85,247,0.12))"
        hero_bdr  = "rgba(99,102,241,0.25)"
        ans_bg    = "rgba(249,250,255,0.95)"
        ans_bdr   = "rgba(99,102,241,0.30)"
        chip_bg   = "rgba(99,102,241,0.10)"
        chip_bdr  = "rgba(99,102,241,0.30)"
        chip_txt  = "#4338ca"
        stat_bg   = "rgba(255,255,255,0.80)"
        stat_bdr  = "rgba(99,102,241,0.18)"
        inp_bg    = "rgba(255,255,255,0.90)"
        inp_bdr   = "rgba(99,102,241,0.25)"
        inp_color = "#1e1b4b"
        scrl_thm  = "rgba(99,102,241,0.35)"
        warn_bg   = "rgba(245,158,11,0.08)"
        warn_bdr  = "rgba(245,158,11,0.30)"
        warn_txt  = "#92400e"
        ok_bg     = "rgba(16,185,129,0.08)"
        ok_bdr    = "rgba(16,185,129,0.30)"
        ok_txt    = "#065f46"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {{ font-family: 'Inter', sans-serif !important; }}

    /* ── Root background ── */
    .stApp {{ background: {bg}; min-height: 100vh; }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {sidebar} !important;
        border-right: 1px solid {border_sb};
    }}

    /* ── Generic card ── */
    .card {{
        background: {card_bg};
        border: 1px solid {card_bdr};
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
    }}

    /* ── Hero banner ── */
    .hero {{
        background: {hero_bg};
        border: 1px solid {hero_bdr};
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        text-align: center;
    }}
    .hero h1 {{ font-size: 2.3rem; font-weight: 700; color: {text_main}; margin: 0; }}
    .hero p  {{ font-size: 1rem;   color: {text_mute}; margin-top: 8px; }}

    /* ── Stat cards ── */
    .stat-grid {{ display: flex; gap: 12px; margin-bottom: 20px; }}
    .stat-card {{
        flex: 1;
        background: {stat_bg};
        border: 1px solid {stat_bdr};
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
    }}
    .stat-card .label {{ font-size: 0.7rem; font-weight: 600; color: {text_mute}; text-transform: uppercase; letter-spacing: .06em; }}
    .stat-card .value {{ font-size: 1.6rem; font-weight: 700; color: {text_main}; line-height: 1.2; margin-top: 4px; }}

    /* ── Answer box ── */
    .answer-box {{
        background: {ans_bg};
        border: 1px solid {ans_bdr};
        border-radius: 14px;
        padding: 20px 24px;
        color: {text_main};
        font-size: 0.97rem;
        line-height: 1.8;
        white-space: pre-wrap;
    }}

    /* ── Source chip ── */
    .source-chip {{
        display: inline-block;
        background: {chip_bg};
        border: 1px solid {chip_bdr};
        border-radius: 8px;
        padding: 2px 9px;
        font-size: 0.75rem;
        color: {chip_txt};
        margin-right: 5px;
        margin-bottom: 4px;
        font-weight: 500;
    }}

    /* ── Timestamp ── */
    .ts {{ font-size: 0.72rem; color: {text_mute}; margin-bottom: 6px; }}

    /* ── Status banners ── */
    .banner-ok {{
        background: {ok_bg}; border: 1px solid {ok_bdr};
        border-radius: 10px; padding: 11px 16px;
        color: {ok_txt}; font-size: 0.88rem;
    }}
    .banner-warn {{
        background: {warn_bg}; border: 1px solid {warn_bdr};
        border-radius: 10px; padding: 11px 16px;
        color: {warn_txt}; font-size: 0.88rem;
    }}

    /* ── Input overrides ── */
    textarea, input[type="text"] {{
        background: {inp_bg} !important;
        border: 1px solid {inp_bdr} !important;
        border-radius: 10px !important;
        color: {inp_color} !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {scrl_thm}; border-radius: 3px; }}

    /* ── Message bubbles ── */
    .user-bubble {{
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        max-width: 82%;
        margin-left: auto;
        margin-bottom: 4px;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(99,102,241,0.25);
    }}
    .ai-label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: {chip_txt};
        margin-bottom: 6px;
        letter-spacing: .04em;
    }}

    /* ── Progress steps ── */
    .progress-step {{
        background: {card_bg};
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 8px 14px;
        margin: 5px 0;
        font-size: 0.85rem;
        color: {text_main};
    }}

    /* ── Footer ── */
    .footer {{ text-align: center; color: {text_foot}; font-size: 0.8rem; padding: 16px 0 8px; }}
    </style>
    """, unsafe_allow_html=True)


inject_css(st.session_state.dark_mode)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def get_chunk_count() -> int:
    """Return number of chunks in the active ChromaDB collection."""
    vs = st.session_state.vectorstore
    if vs is None:
        return 0
    try:
        return vs._collection.count()
    except Exception:
        return 0


def db_exists() -> bool:
    return Path(get_chroma_path()).exists()


def format_ts(ts: str) -> str:
    """Format ISO timestamp into human-readable HH:MM."""
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M")
    except Exception:
        return ts


def export_chat_json() -> str:
    """Serialise chat history to a formatted JSON string."""
    return json.dumps(
        [
            {
                "timestamp": e.get("timestamp", ""),
                "question": e["question"],
                "answer": e["answer"],
                "sources": [
                    {"page": s["page"], "content": s["content"][:300]}
                    for s in e.get("sources", [])
                ],
            }
            for e in st.session_state.chat_history
        ],
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Branding ────────────────────────────────────────────────────────────────
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown("## ⚖️ AI Lawyer")
    with col_toggle:
        # Dark / light theme toggle
        moon = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(moon, help="Toggle dark/light mode", key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    # ── Document Upload ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        label="PDF upload",
        type=["pdf"],
        help="Supports contracts, NDAs, agreements, policies, and more.",
        label_visibility="collapsed",
    )

    if st.button("⚡ Process Document", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.warning("⚠️ Please upload a PDF first.")
        else:
            progress_placeholder = st.empty()
            try:
                ensure_data_directory()

                def show_step(msg: str):
                    progress_placeholder.markdown(
                        f'<div class="progress-step">⏳ {msg}</div>',
                        unsafe_allow_html=True,
                    )

                with st.spinner("Processing document…"):
                    show_step("Saving uploaded file…")
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False, dir="data"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    show_step("Parsing PDF pages…")
                    # ingest_pdf handles: load → split → embed → store
                    vs = ingest_pdf(tmp_path, collection_name="legal_docs")

                    show_step("Building RAG chain…")
                    chain = build_qa_chain(vs)

                # Persist to session state
                st.session_state.vectorstore = vs
                st.session_state.qa_chain    = chain
                st.session_state.doc_name    = uploaded_file.name
                st.session_state.doc_chunks  = get_chunk_count()
                st.session_state.chat_history = []   # fresh history for new doc

                Path(tmp_path).unlink(missing_ok=True)  # clean up temp file
                progress_placeholder.empty()
                st.success(f"✅ **{uploaded_file.name}** ready!")
            except ValueError as exc:
                progress_placeholder.empty()
                st.error(f"❌ Configuration error:\n\n{exc}")
            except FileNotFoundError as exc:
                progress_placeholder.empty()
                st.error(f"❌ File error:\n\n{exc}")
            except Exception as exc:
                progress_placeholder.empty()
                st.error(f"❌ Ingestion failed:\n\n{exc}")

    # ── Load existing DB ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗄️ Load Existing DB")
    col_l, col_s = st.columns([3, 1])
    with col_l:
        load_btn = st.button("Load ChromaDB", use_container_width=True)
    with col_s:
        db_icon = "🟢" if db_exists() else "🔴"
        st.markdown(f"<div style='text-align:center;font-size:1.4rem;padding-top:6px'>{db_icon}</div>", unsafe_allow_html=True)

    if load_btn:
        if not db_exists():
            st.error("No ChromaDB found. Ingest a document first.")
        else:
            try:
                with st.spinner("Loading vectorstore…"):
                    vs    = load_vectorstore(collection_name="legal_docs")
                    chain = build_qa_chain(vs)
                st.session_state.vectorstore = vs
                st.session_state.qa_chain    = chain
                st.session_state.doc_name    = "Previously ingested document"
                st.session_state.doc_chunks  = get_chunk_count()
                st.success("✅ Vectorstore loaded!")
            except Exception as exc:
                st.error(f"❌ {exc}")

    # ── Status Panel ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Status")

    # Active document banner
    if st.session_state.doc_name:
        st.markdown(
            f'<div class="banner-ok">📄 <strong>{st.session_state.doc_name}</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-warn">⚠️ No document loaded. Upload a PDF.</div>',
            unsafe_allow_html=True,
        )

    # Stat cards: chunks + model
    provider = get_llm_provider()
    provider_label = "OpenAI" if provider == "openai" else "Ollama (local)"
    chunks = st.session_state.doc_chunks
    history_len = len(st.session_state.chat_history)

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="label">Chunks</div>
            <div class="value">{chunks}</div>
        </div>
        <div class="stat-card">
            <div class="label">Q&A's</div>
            <div class="value">{history_len}</div>
        </div>
    </div>
    <div class="stat-card" style="margin-bottom:4px;">
        <div class="label">🤖 LLM Provider</div>
        <div class="value" style="font-size:1rem;margin-top:4px;">{provider_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Example Queries ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    EXAMPLE_QUERIES = [
        "What are the termination clauses?",
        "What are the payment terms?",
        "Who is responsible for liabilities?",
        "Summarize this agreement.",
        "What are the confidentiality obligations?",
        "What happens in case of a dispute?",
        "What is the governing law and jurisdiction?",
        "Are there any indemnification provisions?",
        "What are the intellectual property rights?",
        "What are the notice requirements?",
    ]
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state["prefill_question"] = q
            st.rerun()

    # ── Actions ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛠️ Actions")

    # Document summary shortcut
    if st.button("📋 Summarize Document", use_container_width=True):
        if st.session_state.qa_chain is None:
            st.warning("Load a document first.")
        else:
            st.session_state["prefill_question"] = (
                "Provide a comprehensive structured summary of this entire legal document. "
                "Include: parties involved, purpose, key obligations, important dates/terms, "
                "and any notable clauses."
            )
            st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Download Q&A as JSON
    if st.session_state.chat_history:
        st.download_button(
            label="⬇️ Export Q&A (JSON)",
            data=export_chat_json(),
            file_name=f"ai_lawyer_qa_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚖️ AI Lawyer</h1>
    <p>Upload a legal document and ask anything — contracts, NDAs, agreements, policies &amp; more.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row (top of main area) ──────────────────────────────────────────────
if st.session_state.doc_name:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📄 Document", st.session_state.doc_name[:22] + "…"
                  if len(st.session_state.doc_name) > 22 else st.session_state.doc_name)
    with c2:
        st.metric("🧩 Chunks Indexed", st.session_state.doc_chunks)
    with c3:
        st.metric("💬 Questions Asked", len(st.session_state.chat_history))
    with c4:
        st.metric("🤖 Model", "OpenAI" if get_llm_provider() == "openai" else "Ollama")
    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")

    for entry in st.session_state.chat_history:
        ts_label = format_ts(entry.get("timestamp", ""))

        # ── User bubble ─────────────────────────────────────────────────────────
        right_col, _ = st.columns([5, 1])
        with right_col:
            st.markdown(
                f'<div class="ts" style="text-align:right">🙋 You  ·  {ts_label}</div>'
                f'<div class="user-bubble">{entry["question"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── AI response ─────────────────────────────────────────────────────────
        _, left_col = st.columns([0, 10])    # full width for AI response
        with st.container():
            st.markdown(
                f'<div class="ts">⚖️ AI Lawyer  ·  {ts_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="answer-box">{entry["answer"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Source passages ─────────────────────────────────────────────────
            sources = entry.get("sources", [])
            if sources:
                with st.expander(
                    f"📄 View {len(sources)} source passage{'s' if len(sources) > 1 else ''}",
                    expanded=False,
                ):
                    for i, src in enumerate(sources, 1):
                        page_label = src["page"]
                        if isinstance(page_label, int):
                            page_label += 1  # 0-indexed → human-friendly

                        # Source metadata chips
                        st.markdown(
                            f'<span class="source-chip">📑 {src["source"]}</span>'
                            f'<span class="source-chip">📃 Page {page_label}</span>'
                            f'<span class="source-chip">Passage {i}</span>',
                            unsafe_allow_html=True,
                        )
                        # Content block
                        st.code(src["content"], language="text")
                        if i < len(sources):
                            st.divider()
            else:
                st.caption("ℹ️ No source passages were retrieved for this answer.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  QUESTION INPUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔍 Ask a Question")

# Consume any prefill set by sidebar buttons
default_question = st.session_state.pop("prefill_question", "")

question = st.text_area(
    label="Legal question",
    value=default_question,
    placeholder="e.g. What are the termination clauses in this contract?",
    height=110,
    label_visibility="collapsed",
    key="question_input",
)

btn_col, hint_col = st.columns([1, 7])
with btn_col:
    ask_btn = st.button("Ask ⚡", type="primary", use_container_width=True)
with hint_col:
    if not st.session_state.doc_name:
        st.markdown(
            "<span style='color:#f59e0b;font-size:0.87rem'>⚠️ Upload and process a document first.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span style='color:#6ee7b7;font-size:0.87rem'>✅ Ready — ask anything about <strong>{st.session_state.doc_name}</strong></span>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN THE RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
if ask_btn:
    if not question.strip():
        st.warning("⚠️ Please enter a question before clicking Ask.")
    elif st.session_state.qa_chain is None:
        st.error(
            "❌ No document loaded. Upload a PDF in the sidebar and click "
            "**⚡ Process Document** first."
        )
    else:
        with st.spinner("🔎 Retrieving relevant passages and generating answer…"):
            try:
                result = answer_question(st.session_state.qa_chain, question)

                # Append to chat history with ISO timestamp
                st.session_state.chat_history.append(
                    {
                        "question": question.strip(),
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                st.rerun()

            except ConnectionError as exc:
                st.error(
                    f"❌ Could not connect to the LLM provider.\n\n"
                    f"Check your API key or Ollama server. Details: {exc}"
                )
            except Exception as exc:
                st.error(
                    f"❌ An error occurred while generating the answer:\n\n{exc}\n\n"
                    "Try rephrasing your question or re-processing the document."
                )

# ═══════════════════════════════════════════════════════════════════════════════
#  EMPTY STATE — Getting Started Guide
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.chat_history and not st.session_state.doc_name:
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="card" style="text-align:center">
            <div style="font-size:2.2rem">📤</div>
            <h4 style="margin:10px 0 6px">Step 1</h4>
            <p style="font-size:0.88rem;color:#94a3b8">Upload a PDF in the sidebar — contracts, NDAs, policies, or any legal document.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card" style="text-align:center">
            <div style="font-size:2.2rem">⚡</div>
            <h4 style="margin:10px 0 6px">Step 2</h4>
            <p style="font-size:0.88rem;color:#94a3b8">Click <strong>Process Document</strong> to parse, chunk, and embed the text into ChromaDB.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="card" style="text-align:center">
            <div style="font-size:2.2rem">💬</div>
            <h4 style="margin:10px 0 6px">Step 3</h4>
            <p style="font-size:0.88rem;color:#94a3b8">Ask natural-language questions. Click an example query in the sidebar to get started fast.</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='footer'>⚖️ AI Lawyer — For informational purposes only. "
    "This is <strong>not</strong> legal advice. Always consult a qualified attorney.</div>",
    unsafe_allow_html=True,
)

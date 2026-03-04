# ⚖️ AI Lawyer — Legal Document Q&A with RAG

A production-quality **Retrieval-Augmented Generation (RAG)** application that lets you upload legal documents (contracts, NDAs, agreements, policies) and ask natural-language questions about them. Powered by **LangChain**, **ChromaDB**, and **OpenAI / Ollama**.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION PIPELINE                  │
│                                                                   │
│  PDF Upload → PyPDFLoader → RecursiveCharacterTextSplitter       │
│      → Embeddings (OpenAI / HuggingFace) → ChromaDB             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ persist to disk
                               ▼
                         [ ChromaDB ]
                               │
                               │ similarity search (MMR)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Q&A PIPELINE (RAG)                          │
│                                                                   │
│  User Question → Embed Query → Retrieve top-k chunks             │
│      → Build Prompt (context + question) → LLM → Answer         │
└─────────────────────────────────────────────────────────────────┘
```

| Pipeline | Technology |
|---|---|
| PDF Parsing | `PyPDFLoader` (LangChain / pypdf) |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `text-embedding-ada-002` (OpenAI) or `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB (persistent, local) |
| Retrieval | MMR (Maximum Marginal Relevance) — balanced relevance + diversity |
| LLM | `gpt-3.5-turbo` / `gpt-4o` (OpenAI) or any Ollama model |
| Interface | Streamlit |

---

## 📁 Project Structure

```
RAG_Project/
│
├── app.py              # Streamlit web interface
├── ingest.py           # Document ingestion pipeline (+ CLI)
├── rag_pipeline.py     # Retriever + LLM Q&A chain
├── utils.py            # Shared config, logging, helpers
│
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
│
├── data/               # Uploaded PDFs stored here
└── chroma_db/          # ChromaDB persistence (auto-created)
```

---

## 🚀 Setup & Installation

### 1. Clone / enter the project directory
```bash
cd /path/to/RAG_Project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Open .env in your editor and set OPENAI_API_KEY
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

1. **Upload a PDF** in the sidebar → click **Process Document**
2. **Ask questions** in the main area, or pick an example query from the sidebar
3. View the AI answer and expand **📄 Source Passages** to see what the model used

---

## 🖥️ CLI Usage (Ingest Only)

```bash
# Ingest a PDF from the command line
python ingest.py --file data/contract.pdf

# Specify a custom collection name
python ingest.py --file data/nda.pdf --collection nda_docs
```

---

## 💬 Example Legal Queries

| Query | What the AI extracts |
|---|---|
| "What are the termination clauses?" | Termination conditions, notice periods |
| "What are the payment terms?" | Payment schedules, late fees, invoicing |
| "Who is responsible for liabilities?" | Indemnification, liability caps |
| "Summarize this agreement." | Plain-English summary of the document |
| "What are the confidentiality obligations?" | NDA scope, duration, exceptions |
| "What governs this contract?" | Governing law, jurisdiction, dispute resolution |
| "Are there any indemnification provisions?" | Indemnity clauses and conditions |

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Your OpenAI secret key |
| `OPENAI_CHAT_MODEL` | `gpt-3.5-turbo` | Chat model name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_CHAT_MODEL` | `llama3` | Ollama model name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector store location |

---

## 🆓 Using Free / Local Models (No API Key Required)

Set in your `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_CHAT_MODEL=mistral    # or llama3, phi3, gemma2
```

Install and start Ollama:
```bash
# Install: https://ollama.com
ollama pull mistral
ollama serve
```

HuggingFace embeddings (`all-MiniLM-L6-v2`) are used automatically — fully offline.

---

## 🌟 Bonus Feature Ideas

| Feature | Implementation hint |
|---|---|
| **Clause extraction** | Prompt engineering to identify specific clause types |
| **Risk detection** | Zero-shot classify clauses as high/medium/low risk |
| **Document summarization** | `MapReduceDocumentsChain` for long documents |
| **Multi-document Q&A** | Separate ChromaDB collections per document |
| **Highlighted source text** | Use `st.markdown` with highlighted HTML spans |
| **Export Q&A report** | `st.download_button` with JSON / PDF export |
| **Table of clauses** | Parse headings with regex + display as sidebar nav |

---

## ⚠️ Disclaimer

> This application is for **informational and educational purposes only**.  
> It does not constitute legal advice. Always consult a qualified attorney for legal matters.

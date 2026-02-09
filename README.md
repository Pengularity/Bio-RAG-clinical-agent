# Bio-RAG: Agentic Clinical LLM & CDSS Engine

**Local-First Retrieval-Augmented Generation with Hybrid Search & Cross-Encoder Reranking.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Core%20%7C%20Community%20%7C%20Chroma-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com/)
[![Chroma](https://img.shields.io/badge/ChromaDB-Vector%20Store-FFCC00?logo=chroma&logoColor=black)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.1%20%7C%20BioMistral-000000?logo=ollama&logoColor=white)](https://ollama.ai/)
[![FlashRank](https://img.shields.io/badge/FlashRank-Cross--Encoder%20Reranker-10B981)](https://github.com/PrithivirajDamodaran/FlashRank)

*Medical NLP · RAG · LLM · Agentic Workflow · Privacy-Preserving (GDPR/HDS) · Local Inference*

---

## The Engineering Challenge

Most RAG setups struggle when the question hinges on fine distinctions—e.g. NYHA II vs III, or a specific HbA1c cutoff. Pure semantic search often returns “close enough” chunks that aren’t actually the right guideline. This project tries to fix that by combining keyword search (BM25) and vector search, then reranking the results so the model gets the most relevant passages.

Everything runs on your machine (Ollama + local embeddings). No guidelines or queries are sent to the cloud. You upload PDFs, ask questions, and get answers tied to the retrieved snippets and their scores—useful as a local prototype for clinical decision support or guideline Q&A.

> **Disclaimer:** This project is for research and prototyping only. Do not use for clinical decisions.

---

## Technical Architecture (V2)

### Two-Stage Retrieval Pipeline

| Stage | Role | Implementation |
|-------|------|----------------|
| **Stage 1 — Recall** | Broad candidate retrieval | **Hybrid Search** via custom `EnsembleRetriever`: BM25 (exact medical acronyms, e.g. *HbA1c*, *INR*) + ChromaDB (semantic context). Reciprocal Rank Fusion (RRF) merges rankings. |
| **Stage 2 — Precision** | Noise filtering & evidence focus | **Reranking** with FlashRank (cross-encoder, ONNX). Top-*k* candidates are rescored so only high-relevance chunks reach the LLM. |

This pipeline is implemented in **`src/app.py`**: ingestion → chunking → hybrid retrieval → rerank → prompt → local LLM response.

### Tech Stack

- **Python** · **LangChain** (Chroma, HuggingFace Embeddings, Ollama)
- **Ollama** — Llama 3.1 / BioMistral (local LLM)
- **Streamlit** — Chat UI and document upload
- **FlashRank** — Cross-encoder reranker (e.g. `ms-marco-MiniLM-L-12-v2`)

**Privacy-first:** The stack runs **fully on-premises** (e.g. single GPU such as RTX 3090). No patient or guideline data leaves the machine.

---

## Features

- **Multi-Document Ingestion** — PDF upload and chunking; persistent vector store (ChromaDB).
- **Strict Evidence-Based Mode** — Two-stage retrieval + reranking reduces irrelevant context and helps curb hallucinations.
- **Explainable AI (XAI)** — UI exposes relevance scores and source pages for traceability and audit.

---

## Project Roadmap

| Status | Version | Description |
|--------|---------|-------------|
| ✅ | **v1.0** | MVP — Basic semantic search with ChromaDB & Llama 3. |
| ✅ | **v2.0** | Production precision layer — Hybrid search (BM25 + vector) & reranking for clinical guidelines. *(Current.)* |
| ✅ | **v3.0** | Scientific benchmarking — RAGAS integration (Faithfulness, Answer Relevancy, Context Precision); `temperature=0` for reproducible scores. |
| ⬜ | **v4.0** | Multimodal agent — Vision (Llava/ColPali) for medical imaging alongside text guidelines. |

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running (e.g. `ollama run llama3.1` or `ollama run biomistral`)
- (Optional) GPU for embeddings and reranker

### Setup

```bash
# Clone and enter project
cd local-medical-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run src/app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`). Upload PDFs via the sidebar, then ask clinical or technical questions; answers are grounded in retrieved chunks and reranked evidence.

### Optional: Pre-build Vector DB (CLI)

```bash
# Place PDFs in data/ then run
python src/ingest.py
```

### Optional: RAGAS evaluation (V3)

Benchmark Naïve vs Advanced RAG with Faithfulness, Answer Relevancy, and Context Precision. See **scripts/README.md** for usage and prerequisites.

```bash
python scripts/evaluate_rag.py --mode both
```

---

## Repository Layout

```
local-medical-rag/
├── src/
│   ├── app.py      # Streamlit app, two-stage retrieval, reranking, chat
│   ├── ingest.py   # PDF ingestion and ChromaDB indexing
│   └── query.py    # CLI query helper
├── scripts/
│   ├── evaluate_rag.py   # RAGAS benchmark (Naïve vs Advanced RAG)
│   └── README.md
├── data/           # Put PDFs here (*.pdf gitignored)
├── requirements.txt
├── LICENSE         # MIT
└── README.md
```

---

*Bio-RAG — Local-first clinical RAG for evidence-based decision support.*

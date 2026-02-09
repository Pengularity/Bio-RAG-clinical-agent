# Scripts

## `evaluate_rag.py` — RAGAS Benchmark (V3)

Benchmarks the Bio-RAG pipeline (Naïve vs Advanced) using **RAGAS** with a **local judge** (Ollama/Llama 3.1). No OpenAI.

### Libraries Used (and why)

| Library | Purpose |
|--------|--------|
| **ragas** | Evaluation framework: Faithfulness, Answer Relevancy, Context Precision. Uses an LLM as “judge” and (for relevancy) embeddings; we pass Ollama + HuggingFace so nothing leaves your machine. |
| **langchain_ollama** | ChatOllama: used both to generate RAG answers and as the RAGAS judge LLM. |
| **langchain_huggingface** | HuggingFaceEmbeddings: used for retrieval (Chroma) and for RAGAS Answer Relevancy (cosine similarity between question and “what the answer is about”). |
| **flashrank** | Cross-encoder reranker for Advanced RAG (same as `src/app.py`). |
| **langchain_chroma**, **langchain_community**, **langchain_text_splitters**, **pypdf**, **rank_bm25** | Same as main app: load PDFs, split, vector store, BM25, hybrid retriever. |

Install everything (including RAGAS):

```bash
pip install -r requirements.txt
```

### Prerequisites

- **Ollama** running with `llama3.1` (e.g. `ollama run llama3.1`).
- At least one PDF in `data/` (e.g. WHO Hypertension guideline). The script loads all PDFs from `data/`, chunks them, and builds the same hybrid retriever as the app.

### Usage

From the **project root**:

```bash
# Compare both pipelines and print RAGAS scores
python scripts/evaluate_rag.py --mode both

# Only Naïve RAG (no reranker)
python scripts/evaluate_rag.py --mode naive

# Only Advanced RAG (hybrid + rerank)
python scripts/evaluate_rag.py --mode advanced

# Custom data folder
python scripts/evaluate_rag.py --mode both --data-dir /path/to/pdfs
```

### Ground truth

The script uses a small built-in set of 5 Q&A pairs (WHO Hypertension theme). To evaluate on your own questions/answers, edit `GROUND_TRUTH_QA` in `scripts/evaluate_rag.py`.

### Reproducibility and Faithfulness

- **RAGAS judge** uses **`temperature=0`** so evaluation scores are reproducible.
- **Answer generation** uses **`temperature=0.1`** and a **stricter prompt** (only explicit context, no inference, prefer quoting) to improve Faithfulness scores while limiting drift.
- **Max response length** is capped (`num_predict=512`) so answers stay focused on the context.

### Output

- RAGAS scores for **Faithfulness**, **Answer Relevancy**, and **Context Precision** (each 0–1).
- If `--mode both`, a short comparison (Advanced vs Naïve) is printed at the end.

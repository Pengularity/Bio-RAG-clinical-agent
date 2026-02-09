#!/usr/bin/env python3
"""
Bio-RAG V3: Quantitative Evaluation with RAGAS
==============================================

This script benchmarks our Hybrid + Rerank pipeline using the RAGAS framework.
It compares "Naïve RAG" (retrieval only) vs "Advanced RAG" (retrieval + reranking).

THE RAGAS TRIAD: Why These Three Metrics?
------------------------------------------

RAGAS evaluates your RAG system along three axes. Think of them as answering:

  1. FAITHFULNESS (Answer ↔ Context)
     "Does the model's answer stick to what was actually in the retrieved chunks?"
     - The metric uses an LLM as a 'judge': it extracts every claim from the answer,
       then checks whether each claim can be inferred from the retrieved context.
     - Score = (number of supported claims) / (total claims). Range: 0 to 1.
     - Low faithfulness = hallucinations or unsupported statements.
     - Why it matters in MedTech: You want answers grounded in guidelines, not invented.

  2. ANSWER RELEVANCY (Question ↔ Answer)
     "Does the answer actually address the user's question?"
     - The judge generates several alternative questions that the answer could answer.
       It then compares these (via embeddings) to the original question.
     - High similarity → the answer is on-topic. Low → generic, redundant, or off-topic.
     - Uses embeddings to compute cosine similarity between question and "what the answer
       is about". So we need a local embeddings model (e.g. HuggingFace) for this metric.
     - Why it matters: Clinicians ask specific questions; answers must be relevant.

  3. CONTEXT PRECISION (Question ↔ Retrieved Contexts)
     "Did the retriever put the most relevant chunks at the top?"
     - Measures whether relevant passages are ranked higher than irrelevant ones.
     - Uses the ground_truth (reference) answer: chunks that help produce a correct
       answer are considered "relevant". The metric rewards ranking those chunks first.
     - Score reflects precision-at-k: are the top-k contexts the right ones?
     - Why it matters: Reranking should improve this; we expect Advanced RAG > Naïve RAG.

Summary:
  - Faithfulness  → Is the answer grounded in the context? (no hallucination)
  - Relevancy     → Is the answer on-topic? (embeddings + LLM judge)
  - Precision     → Did retrieval rank the best chunks first? (ground truth needed)

All three use an LLM as judge (we use Ollama/Llama 3.1). Answer Relevancy also uses
embeddings (we use HuggingFace all-MiniLM-L6-v2). No OpenAI required.
"""

import os
import sys
import argparse
import warnings

# Add project root so we can load PDFs from data/ and use consistent paths when run as:
#   python scripts/evaluate_rag.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- LangChain: document loading, splitting, vector store, retrievers ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- FlashRank: reranker for Advanced RAG (same as app.py) ---
from flashrank import Ranker, RerankRequest

# --- RAGAS: evaluation framework. We pass our own LLM and embeddings (Ollama + HuggingFace). ---
# Use legacy metrics (ragas.metrics): they accept LangChain LLM/embeddings via evaluate().
# The "collections" metrics require InstructorBaseRagasLLM and do not support Ollama directly.
# Suppress deprecation warning for these imports until RAGAS supports LangChain/Ollama in collections.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*ragas\.metrics.*deprecated.*",
    )
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from ragas import EvaluationDataset, SingleTurnSample


# =============================================================================
# CONFIGURATION (paths and model names; same as app/ingest for reproducibility)
# =============================================================================

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
# Where we look for PDFs (e.g. WHO Hypertension guideline). Must exist and contain at least one PDF.
EVAL_LLM_MODEL = "llama3.1"  # RAGAS judge + answer generation; keep same as app for fair comparison
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVE = 15   # How many chunks the hybrid retriever returns before (optional) rerank
TOP_K_AFTER_RERANK = 5  # How many chunks we send to the LLM after reranking (Advanced RAG)
TOP_K_NAIVE = 5       # How many chunks we send to the LLM in Naïve RAG (no reranker)
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_CACHE = os.path.join(PROJECT_ROOT, "opt", "flashrank")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# ENSEMBLE RETRIEVER (Reciprocal Rank Fusion)
# Same logic as src/app.py: combine BM25 + vector search with RRF so we reuse
# the exact same retrieval pipeline for evaluation.
# =============================================================================

class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion (RRF)."""

    retrievers: list
    weights: list
    c: int = 60

    def _get_relevant_documents(self, query: str) -> list[Document]:
        from collections import defaultdict
        rrf_scores = defaultdict(float)
        doc_by_key = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs, start=1):
                key = (doc.page_content[:500], str(doc.metadata)[:200])
                key_str = str(key)
                rrf_scores[key_str] += weight * (1.0 / (self.c + rank))
                if key_str not in doc_by_key:
                    doc_by_key[key_str] = doc
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_by_key[k] for k in sorted_keys]


# =============================================================================
# GROUND TRUTH: Evaluation Dataset (WHO Hypertension theme)
# These are ideal Q&A pairs that a good RAG should approximate when the
# underlying PDF (e.g. WHO guideline) contains this information. If your PDF
# is different, replace with questions/answers that match your documents.
# =============================================================================

GROUND_TRUTH_QA = [
    {
        "question": "What are the blood pressure thresholds for diagnosing hypertension in adults?",
        "ground_truth": "Hypertension in adults is typically defined as systolic blood pressure ≥140 mmHg and/or diastolic blood pressure ≥90 mmHg, based on repeated measurements.",
    },
    {
        "question": "What first-line pharmacological treatments are recommended for hypertension?",
        "ground_truth": "First-line pharmacological options for hypertension often include ACE inhibitors, angiotensin receptor blockers (ARBs), calcium channel blockers (CCBs), or thiazide diuretics, depending on guidelines and patient profile.",
    },
    {
        "question": "What lifestyle modifications are recommended for hypertensive patients?",
        "ground_truth": "Lifestyle modifications include reducing salt intake, increasing physical activity, maintaining a healthy weight, limiting alcohol, and eating a diet rich in fruits and vegetables (e.g. DASH-style diet).",
    },
    {
        "question": "When should antihypertensive treatment be initiated in adults?",
        "ground_truth": "Treatment is typically initiated when blood pressure is consistently at or above the threshold (e.g. ≥140/90 mmHg) despite lifestyle measures, or earlier in high-risk patients depending on guidelines.",
    },
    {
        "question": "What is the target blood pressure for most adults with hypertension?",
        "ground_truth": "For most adults, the target is often below 140/90 mmHg; some guidelines recommend lower targets (e.g. <130/80 mmHg) for high-risk or diabetic patients.",
    },
    # {
    #     "question": "What is the U-Net architecture used for?",
    #     "ground_truth": "U-Net is a convolutional network for biomedical image segmentation.",
    # },
    # {
    #     "question": "What are skip connections in U-Net?",
    #     "ground_truth": "Skip connections concatenate feature maps from the encoder to the decoder to preserve spatial detail.",
    # },
]


# =============================================================================
# BUILD RETRIEVER FROM data/ PDFs
# Loads all PDFs, splits with same chunk size/overlap as app, builds Chroma + BM25
# and EnsembleRetriever. This mirrors what the Streamlit app does when you upload files.
# =============================================================================

def build_retriever_from_data_folder():
    """Load PDFs from DATA_PATH, chunk them, and return our Hybrid (BM25 + Chroma) retriever."""
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Data folder not found: {DATA_PATH}. Create it and add at least one PDF.")
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files in {DATA_PATH}. Add e.g. a WHO Hypertension PDF.")

    # Load all pages from all PDFs into LangChain Document objects
    all_docs = []
    for f in pdf_files:
        path = os.path.join(DATA_PATH, f)
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = f
        all_docs.extend(docs)

    # Split into chunks (same as app.py and ingest.py for comparable retrieval)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(all_docs)

    # Embeddings: used by Chroma for vector search. Same model as app for consistency.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Vector store (in-memory for this script; we don't persist the eval DB)
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

    # BM25 retriever (keyword-based; good for exact terms like "HbA1c", "mmHg")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_RETRIEVE

    # Vector retriever
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})

    # Hybrid: RRF of BM25 + vector (same weights as app)
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6],
    )
    return ensemble, chunks  # chunks kept in case we need them later


# =============================================================================
# RERANKER (FlashRank) for Advanced RAG
# =============================================================================

def get_reranker():
    """Load FlashRank cross-encoder for reranking (same as app.py)."""
    return Ranker(model_name=FLASHRANK_MODEL, cache_dir=FLASHRANK_CACHE)


# =============================================================================
# ANSWER GENERATION
# We use the same prompt style as app.py: strict context-only, no hallucination.
# =============================================================================

PROMPT_TEMPLATE = """Use ONLY the information explicitly stated in the context below. Do not add, infer, or paraphrase beyond what is written. If the context does not contain the answer, say "I don't know." Prefer quoting or staying very close to the context wording. Focus on clinical thresholds and specific scores (e.g., NYHA, HbA1c, blood pressure) when present.

CONTEXT:
{context}

QUESTION: {question}
"""

# Slightly higher temperature for generation can improve faithfulness (judge scores answers
# as better grounded). Judge LLM stays at temperature=0 for reproducibility.
EVAL_GENERATION_TEMPERATURE = 0.1
EVAL_MAX_TOKENS = 512


def generate_answer(query: str, context_texts: list[str], model_name: str = EVAL_LLM_MODEL) -> str:
    """Call Ollama with the given context and question; return the model's answer string."""
    context = "\n\n".join(context_texts)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format(context=context, question=query)
    llm = ChatOllama(
        model=model_name,
        temperature=EVAL_GENERATION_TEMPERATURE,
        num_predict=EVAL_MAX_TOKENS,
    )
    response = llm.invoke(formatted)
    return response.content if hasattr(response, "content") else str(response)


# =============================================================================
# RUN RAG AND COLLECT SAMPLES FOR RAGAS
# RAGAS expects each sample to have: user_input, retrieved_contexts, response, reference.
# - user_input: the question
# - retrieved_contexts: list of strings (the chunks we passed to the LLM)
# - response: the generated answer
# - reference: ground truth answer (for context precision and optional other metrics)
# =============================================================================

def run_naive_rag(retriever, model_name: str = EVAL_LLM_MODEL):
    """
    Naïve RAG: retrieve TOP_K_NAIVE chunks (no reranking), then generate answer.
    Returns a list of dicts suitable for RAGAS EvaluationDataset.from_list().
    """
    samples = []
    for item in GROUND_TRUTH_QA:
        question = item["question"]
        reference = item["ground_truth"]

        # Retrieve: hybrid returns many chunks; we take the first TOP_K_NAIVE
        docs = retriever.invoke(question)
        context_texts = [d.page_content for d in docs[:TOP_K_NAIVE]]

        # Generate answer using only those chunks
        response = generate_answer(question, context_texts, model_name)

        samples.append({
            "user_input": question,
            "retrieved_contexts": context_texts,
            "response": response,
            "reference": reference,
        })
    return samples


def run_advanced_rag(retriever, ranker, model_name: str = EVAL_LLM_MODEL):
    """
    Advanced RAG: retrieve TOP_K_RETRIEVE chunks, rerank with FlashRank, take top
    TOP_K_AFTER_RERANK, then generate answer. Same pipeline as src/app.py.
    """
    samples = []
    for item in GROUND_TRUTH_QA:
        question = item["question"]
        reference = item["ground_truth"]

        # Stage 1: Hybrid retrieval
        docs = retriever.invoke(question)

        # Stage 2: Rerank with FlashRank (cross-encoder)
        passages = [
            {"id": i, "text": d.page_content, "meta": d.metadata}
            for i, d in enumerate(docs)
        ]
        rerank_request = RerankRequest(query=question, passages=passages)
        rerank_results = ranker.rerank(rerank_request)
        top = rerank_results[:TOP_K_AFTER_RERANK]
        context_texts = [r["text"] for r in top]

        # Stage 3: Generate answer
        response = generate_answer(question, context_texts, model_name)

        samples.append({
            "user_input": question,
            "retrieved_contexts": context_texts,
            "response": response,
            "reference": reference,
        })
    return samples


# =============================================================================
# RAGAS EVALUATION WITH LOCAL MODELS
# We pass ChatOllama as the judge LLM and HuggingFace embeddings. RAGAS wraps
# them via LangchainLLMWrapper / LangchainEmbeddingsWrapper internally when you
# pass LangChain objects to evaluate().
# =============================================================================

def run_ragas_evaluation(samples: list, mode_label: str):
    """
    Build RAGAS EvaluationDataset from samples and run the three metrics:
    Faithfulness, Answer Relevancy, Context Precision.
    """
    # Convert list of dicts to RAGAS dataset. Newer RAGAS uses SingleTurnSample + EvaluationDataset(samples=...).
    try:
        ragas_samples = [
            SingleTurnSample(
                user_input=s["user_input"],
                retrieved_contexts=s["retrieved_contexts"],
                response=s["response"],
                reference=s["reference"],
            )
            for s in samples
        ]
        dataset = EvaluationDataset(samples=ragas_samples)
    except (TypeError, AttributeError):
        # Fallback: older RAGAS versions expose EvaluationDataset.from_list(list_of_dicts)
        dataset = EvaluationDataset.from_list(samples)

    # Local judge and embeddings: no OpenAI. RAGAS will use these for all three metrics.
    llm = ChatOllama(model=EVAL_LLM_MODEL, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Run evaluation. Faithfulness and Context Precision use the LLM; Answer Relevancy
    # uses both LLM and embeddings (to compare question vs. generated "alternative questions").
    # Legacy metrics are pre-initialised instances; evaluate() injects llm/embeddings into them.
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    # RAGAS returns an EvaluationResult object: result.scores is a list of dicts (per sample),
    # and the aggregated means are in result._repr_dict (metric name -> mean score).
    if getattr(result, "_repr_dict", None):
        scores_dict = result._repr_dict
    elif result.scores:
        n = len(result.scores)
        scores_dict = {k: sum(d[k] for d in result.scores) / n for k in result.scores[0].keys()}
    else:
        scores_dict = {}
    print(f"\n--- RAGAS scores ({mode_label}) ---")
    for name, value in scores_dict.items():
        print(f"  {name}: {value:.4f}")
    return scores_dict


# =============================================================================
# MAIN: Parse mode (naive | advanced | both), build retriever, run RAG, evaluate.
# =============================================================================

def main():
    global DATA_PATH

    parser = argparse.ArgumentParser(
        description="Evaluate Bio-RAG pipeline with RAGAS (Naïve vs Advanced RAG)."
    )
    parser.add_argument(
        "--mode",
        choices=["naive", "advanced", "both"],
        default="both",
        help="Which pipeline to evaluate: naive (no reranker), advanced (with reranker), or both.",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_PATH,
        help="Folder containing PDFs (e.g. WHO Hypertension guideline).",
    )
    args = parser.parse_args()

    DATA_PATH = args.data_dir

    print("Building retriever from PDFs in:", DATA_PATH)
    retriever, _ = build_retriever_from_data_folder()
    print("Retriever ready (Hybrid: BM25 + Chroma).")

    results = {}

    if args.mode in ("naive", "both"):
        print("\nRunning Naïve RAG (no reranker)...")
        naive_samples = run_naive_rag(retriever)
        results["naive"] = run_ragas_evaluation(naive_samples, "Naïve RAG")

    if args.mode in ("advanced", "both"):
        print("\nLoading reranker (FlashRank)...")
        ranker = get_reranker()
        print("Running Advanced RAG (hybrid + rerank)...")
        advanced_samples = run_advanced_rag(retriever, ranker)
        results["advanced"] = run_ragas_evaluation(advanced_samples, "Advanced RAG")

    if args.mode == "both" and "naive" in results and "advanced" in results:
        print("\n--- Comparison (Advanced vs Naïve) ---")
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            n = results["naive"].get(metric, 0)
            a = results["advanced"].get(metric, 0)
            diff = a - n
            print(f"  {metric}: Naïve={n:.4f}  Advanced={a:.4f}  (Δ = {diff:+.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()

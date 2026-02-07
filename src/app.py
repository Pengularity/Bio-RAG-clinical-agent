import streamlit as st
import os
import tempfile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# V2 Additions: Hybrid Search and Reranking
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest


class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers using Reciprocal Rank Fusion (RRF)."""

    retrievers: list
    weights: list[float]
    c: int = 60  # RRF constant

    def _get_relevant_documents(self, query: str) -> list[Document]:
        from collections import defaultdict
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_by_key: dict[str, Document] = {}

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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bio-RAG V2", page_icon="🧬", layout="wide")

# --- INITIALIZE RERANKER (Stage 2 Model) ---
@st.cache_resource
def load_reranker():
    # A small but powerful model to re-evaluate search results
    return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="opt/flashrank")

ranker = load_reranker()

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.title("⚙️ Bio-RAG V2 Settings")
    uploaded_files = st.file_uploader("Upload Medical PDFs (HAS/WHO)", type=["pdf"], accept_multiple_files=True)
    model_choice = st.selectbox("LLM Model", ["llama3.1", "biomistral"], index=0)
    
    if st.button("Clear Data & History"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.retriever = None
        st.rerun()

st.title("🧬 Advanced Bio-RAG for CDSS")
st.caption("Production-grade pipeline: Hybrid Search + Cross-Encoder Reranking")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "retriever" not in st.session_state: st.session_state.retriever = None

# --- STAGE 1: THE INGESTION ENGINE ---
def process_documents(uploaded_files):
    all_chunks = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    with st.spinner("Analyzing medical literature..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs: d.metadata["source"] = uploaded_file.name
            
            # Splitting text for the database
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            os.remove(tmp_path)

        # 1. Vector Database (Semantic Search)
        vector_db = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        
        # 2. BM25 (Keyword Search)
        bm25_retriever = BM25Retriever.from_documents(all_chunks)
        bm25_retriever.k = 15 # Pull top 15 by keyword
        
        # 3. Hybrid Retriever (Combined)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 15})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6] # 40% Keyword, 60% Semantic
        )
        return ensemble_retriever

# --- PROCESSING TRIGGER ---
if uploaded_files and st.session_state.retriever is None:
    st.session_state.retriever = process_documents(uploaded_files)
    st.success(f"✅ {len(uploaded_files)} documents indexed with Hybrid Search.")

# --- CHAT INTERFACE ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask a clinical or technical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    if st.session_state.retriever:
        with st.chat_message("assistant"):
            # A. STAGE 1: HYBRID RETRIEVAL
            with st.spinner("Retrieving candidates..."):
                initial_docs = st.session_state.retriever.invoke(prompt)
            
            # B. STAGE 2: RERANKING (The Filter)
            with st.spinner("Reranking for precision..."):
                passages = [
                    {"id": i, "text": d.page_content, "meta": d.metadata} 
                    for i, d in enumerate(initial_docs)
                ]
                rerank_request = RerankRequest(query=prompt, passages=passages)
                rerank_results = ranker.rerank(rerank_request)
                
                # We take only the top 5 after reranking
                final_docs = rerank_results[:5]
                context = "\n\n".join([r["text"] for r in final_docs])

            # C. GENERATION
            PROMPT_TEMPLATE = """
            Use the strictly provided context to answer. If unsure, say you don't know.
            Focus on clinical thresholds and specific scores (e.g., NYHA, HbA1c).

            CONTEXT:
            {context}

            QUESTION: {question}
            """
            model = ChatOllama(model=model_choice)
            full_prompt = PROMPT_TEMPLATE.format(context=context, question=prompt)
            
            response = model.invoke(full_prompt)
            st.markdown(response.content)
            
            # D. DISPLAY SOURCES
            with st.expander("🔍 Evidence & Reranking Scores"):
                for r in final_docs:
                    st.write(f"**Score: {r['score']:.4f}** | Source: {r['meta']['source']}")
                    st.info(r['text'][:200] + "...")

        st.session_state.messages.append({"role": "assistant", "content": response.content})
    else:
        st.warning("Please upload documents first.")
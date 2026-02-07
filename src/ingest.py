import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/"
DB_PATH = "vector_db"

def create_vector_db():
    """
    Reads PDFs from the data/ folder, splits them into chunks,
    and stores them in a local ChromaDB vector database.
    """
    
    # 1. Check for PDF files
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}' folder. Please put your medical PDFs here.")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_PATH}'. Add a file and retry.")
        return

    # 2. Load Documents
    documents = []
    print(f"Found {len(pdf_files)} PDF(s). Loading...")
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_PATH, pdf_file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
        print(f"   - Loaded: {pdf_file} ({len(docs)} pages)")

    # 3. Split Text (same as app.py so pre-built DB matches in-app behavior)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 4. Initialize Embeddings
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Store in ChromaDB
    print(f"Saving to local Vector DB at '{DB_PATH}'...")
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"Success! Vector Database created with {len(chunks)} knowledge fragments.")

if __name__ == "__main__":
    create_vector_db()
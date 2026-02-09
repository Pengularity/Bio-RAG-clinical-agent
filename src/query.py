import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
DB_PATH = "vector_db"

MODEL_NAME = "llama3.1"
# MODEL_NAME = "qwen2.5-coder:32b"

PROMPT_TEMPLATE = """
You are a specialized medical research assistant.
Answer the question based ONLY on the following context.
If the answer is not in the context, reply: "I cannot find the answer in the provided document."

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
"""

def main():
    # 1. Load the Database
    print("Loading vector database...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # 2. Interactive Loop
    print(f"🤖 Connected to {MODEL_NAME}. Ready to answer questions from your documents.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query_text = input("❓ Ask a question: ")
        if query_text.lower() in ["exit", "quit"]:
            break

        # 3. Retrieval
        # Search the DB for the top 5 most relevant chunks (k=5)
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        # 4. Generation
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        print("\n🤔 Thinking...")
        model = ChatOllama(model=MODEL_NAME)
        response_text = model.invoke(prompt)

        # 5. Output
        print(f"\n💡 Answer:\n{response_text.content}\n")

if __name__ == "__main__":
    main()
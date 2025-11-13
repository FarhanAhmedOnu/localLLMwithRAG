from pdf_utils import extract_text_from_pdf, split_text
from vectorstore import (
    get_embedding_model,
    create_or_load_collection,
    populate_vectorstore,
    retrieve_context
)
from ollama_utils import query_ollama
import os

def run_rag_pipeline(pdf_path, user_query, ollama_model="llama3"):
    """End-to-end RAG pipeline using a PDF and Ollama model."""

    # Ensure the vector store exists
    persist_dir = "vector_store"
    collection = create_or_load_collection(persist_dir)

    # Step 1: Check if this PDF was already embedded
    pdf_name = os.path.basename(pdf_path)
    existing_docs = collection.count()

    if existing_docs == 0:
        print(f"🧾 Indexing {pdf_name}...")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        embedding_model = get_embedding_model()
        populate_vectorstore(collection, chunks, embedding_model)
    else:
        print(f"✅ Reusing existing ChromaDB with {existing_docs} documents.")
        embedding_model = get_embedding_model()

    # Step 2: Retrieve relevant context
    context = retrieve_context(user_query, collection, embedding_model)

    # Step 3: Build prompt (avoid fancy Unicode)
    prompt = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )

    # Step 4: Query Ollama
    answer = query_ollama(ollama_model, prompt)
    return answer

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
    
    # Ensure the vector store directory exists
    persist_dir = "vector_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Get collection
    collection = create_or_load_collection(persist_dir)
    embedding_model = get_embedding_model()

    # Note: PDF processing is now handled by the web UI
    # The web UI manages adding PDFs to the vector store
    
    # Step 1: Retrieve relevant context
    context = retrieve_context(user_query, collection, embedding_model)

    if not context.strip():
        return "I couldn't find relevant information in the documents to answer your question."

    # Step 2: Build prompt
    prompt = (
        "Answer the question using the context below. If the context doesn't contain "
        "relevant information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )

    # Step 3: Query Ollama
    answer = query_ollama(ollama_model, prompt)
    return answer
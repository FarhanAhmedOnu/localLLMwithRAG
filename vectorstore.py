from sentence_transformers import SentenceTransformer
import chromadb

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load sentence-transformer embedding model."""
    return SentenceTransformer(model_name)

def create_or_load_collection(persist_dir="vector_store", name="pdf_docs"):
    """Create or load an existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=name)
    return collection

def populate_vectorstore(collection, chunks, embedding_model):
    """Add text chunks and embeddings to ChromaDB collection."""
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB.")

def retrieve_context(query, collection, embedding_model, top_k=3):
    """Search vector DB for context related to a query."""
    query_emb = embedding_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    context = " ".join(results["documents"][0])
    return context

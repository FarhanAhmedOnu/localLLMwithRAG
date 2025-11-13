import streamlit as st
import os
import tempfile
from rag_pipeline import run_rag_pipeline
from pdf_utils import extract_text_from_pdf, split_text
from vectorstore import (
    get_embedding_model,
    create_or_load_collection,
    populate_vectorstore
)
import chromadb

# Page configuration
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f0f0f0;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    if 'available_pdfs' not in st.session_state:
        st.session_state.available_pdfs = []
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None

def get_available_models():
    """Get available Ollama models."""
    # You can modify this to dynamically fetch models from Ollama
    return ["qwen3:0.6b"]

def check_vector_store_status():
    """Check if vector store has documents."""
    try:
        collection = create_or_load_collection()
        count = collection.count()
        return count > 0
    except:
        return False

def add_pdf_to_vector_store(pdf_path, pdf_name):
    """Add a new PDF to the vector store."""
    try:
        # Extract and process text
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        
        # Get embedding model and collection
        embedding_model = get_embedding_model()
        collection = create_or_load_collection()
        
        # Add to vector store with unique IDs
        existing_count = collection.count()
        embeddings = embedding_model.encode(chunks).tolist()
        ids = [f"{pdf_name}_chunk_{i+existing_count}" for i in range(len(chunks))]
        
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        
        # Update available PDFs
        if pdf_name not in st.session_state.available_pdfs:
            st.session_state.available_pdfs.append(pdf_name)
        
        return True, f"✅ Successfully added '{pdf_name}' with {len(chunks)} chunks"
    except Exception as e:
        return False, f"❌ Error adding PDF: {str(e)}"

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.messages = []

def main():
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header"> RAG PDF Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Ollama Model",
            available_models,
            index=0
        )
        
        st.divider()
        
        # PDF Management
        st.header("📄 PDF Management")
        
        # Upload new PDF
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type="pdf",
            help="Upload a PDF file to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Add to vector store
            if st.button("Add PDF to Knowledge Base"):
                with st.spinner("Processing PDF..."):
                    success, message = add_pdf_to_vector_store(tmp_path, uploaded_file.name)
                    if success:
                        st.success(message)
                        st.session_state.vector_store_initialized = True
                    else:
                        st.error(message)
            
            # Clean up temp file
            os.unlink(tmp_path)
        
        st.divider()
        
        # Vector store status
        st.header("📊 Status")
        has_documents = check_vector_store_status()
        if has_documents:
            collection = create_or_load_collection()
            doc_count = collection.count()
            st.success(f"✅ Knowledge Base: {doc_count} chunks")
            
            # Show available PDFs
            if st.session_state.available_pdfs:
                st.write("**Available Documents:**")
                for pdf in st.session_state.available_pdfs:
                    st.write(f"• {pdf}")
        else:
            st.warning("⚠️ No documents in knowledge base. Upload a PDF to start.")
        
        st.divider()
        
        # Chat controls
        if st.button("Clear Chat History"):
            clear_chat_history()
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Chat")
        
        # Check if vector store is ready
        if not check_vector_store_status():
            st.info(" Please upload a PDF document in the sidebar to start chatting.")
        else:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your PDF..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Use the first available PDF or implement PDF selection logic
                            pdf_path = "uploaded_documents/"  # You might want to modify this
                            response = run_rag_pipeline(
                                pdf_path="1.pdf",  # Adjust based on your needs
                                user_query=prompt,
                                ollama_model=selected_model
                            )
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with col2:
        st.header("Info")
        st.info("""
        **How to use:**
        1. Upload a PDF in the sidebar
        2. Add it to the knowledge base
        3. Start asking questions!
        
        **Features:**
        • Multiple PDF support
        • Customizable models
        • Persistent vector store
        • Real-time chat
        """)

if __name__ == "__main__":
    main()
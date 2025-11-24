# RAG PDF Chat Assistant

A Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents using local LLMs through Ollama.

## Features

- **Document Processing**: Extract and chunk text from PDF files
- **Vector Search**: Store and retrieve document chunks using ChromaDB
- **Local LLM Integration**: Use Ollama models for question answering
- **Web Interface**: User-friendly Streamlit web UI
- **Multiple PDF Support**: Add multiple documents to the knowledge base
- **Persistent Storage**: Vector store persists between sessions

## Technology Stack

- **Backend Framework**: Python
- **PDF Processing**: PyMuPDF (fitz)
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Integration**: Ollama
- **Web Interface**: Streamlit
- **Text Chunking**: Custom overlapping chunking algorithm


## How to Run

First we need to run the ollama server

### Option 1: Command Line Interface (CLI)

Use `main.py` for simple command-line usage:

```bash
python main.py
```

**Configuration in `main.py`:**
```python
pdf_path = "1.pdf"           # Path to your PDF file
question = "What is Perceptron?"  # Your question
model = "qwen3:0.6b"         # Ollama model to use
```

### Option 2: Web Interface

Use `webui.py` for an interactive web experience:

```bash
streamlit run webui.py
```

This will open a web browser with the RAG PDF Chat Assistant interface.


### Using the Web Interface:

1. **Start the application:**
   ```bash
   streamlit run webui.py
   ```

2. **Configure the model:**
   - Select your preferred Ollama model from the sidebar

3. **Upload PDFs:**
   - Use the file uploader in the sidebar to upload PDF documents
   - Click "Add PDF to Knowledge Base" to process and store the document

4. **Chat with your documents:**
   - Type questions in the chat input at the bottom
   - The system will retrieve relevant context and generate answers

5. **Manage the knowledge base:**
   - View status and available documents in the sidebar
   - Clear chat history when needed

### Using the Command Line:

1. **Modify `main.py` with your PDF path and question**
2. **Run the script:**
   ```bash
   python main.py
   ```




### Models
The system uses `qwen3:0.6b` by default, but you can use any model available in Ollama:
- `llama3`
- `mistral`
- `qwen2.5` 
- Or any other supported model

### Text Chunking
Default chunk size: 500 characters with 50-character overlap. Modify in `pdf_utils.py` if needed.

### Vector Store
- Persistent storage in `vector_store/` directory
- Uses `all-MiniLM-L6-v2` for embeddings by default

## Troubleshooting

### Common Issues:

1. **"Ollama not found"**
   - Ensure Ollama is installed and in your PATH
   - Verify Ollama is running: `ollama list`

2. **No response from model**
   - Check if the specified model is pulled: `ollama list`
   - Pull the model if missing: `ollama pull model_name`

3. **PDF processing errors**
   - Ensure PyMuPDF is installed: `pip install pymupdf`
   - Verify PDF file is not corrupted

4. **Vector store issues**
   - Delete `vector_store/` directory to reset
   - Ensure write permissions in the project directory

## Notes

- The first run will be slower as it downloads the embedding model
- Larger PDFs will take longer to process initially
- Chat history is maintained during the web session
- Vector store persists between application restarts

## Customization

You can modify:
- Chunk size and overlap in `pdf_utils.py`
- Embedding model in `vectorstore.py`
- Number of retrieved chunks in `rag_pipeline.py`
- Prompt template in `rag_pipeline.py`

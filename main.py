from rag_pipeline import run_rag_pipeline

if __name__ == "__main__":
    pdf_path = "1.pdf"   # 🔹 Change this to your actual PDF path
    question = " what is Perceptron?"
    model = "qwen3:0.6b"             # 🔹 Or any model available in Ollama

    print("🚀 Running RAG pipeline...\n")
    answer = run_rag_pipeline(pdf_path, question, ollama_model=model)
    print("\n🧠 LLM Answer:\n")
    print(answer)

import subprocess

def query_ollama(model: str, prompt: str) -> str:
    """
    Send a prompt to the Ollama model and return the plain text response.
    Compatible with older Ollama versions (no --json flag).
    """
    cmd = ["ollama", "run", model]

    try:
        # Run Ollama as a subprocess and capture stdout
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        # Send the prompt and close stdin
        process.stdin.write(prompt)
        process.stdin.close()

        # Read output and errors
        output, stderr = process.communicate()

        if stderr.strip():
            print("⚠️ Ollama stderr:", stderr.strip())

        if not output.strip():
            print("⚠️ Warning: No response received from Ollama.")

        return output.strip()

    except FileNotFoundError:
        print("❌ Error: Ollama is not installed or not found in PATH.")
        return ""

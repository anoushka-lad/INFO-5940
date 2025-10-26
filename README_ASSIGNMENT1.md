Assignment 1 — Document Q&A Chatbot

This project is a document-questioning application that ingests PDF, TXT, or MD files, converts them into embeddings, and provides a conversational chatbot interface for asking questions with cited answers.

Key ideas:
- Split long documents into overlapping chunks to preserve context.
- Create vector embeddings (Chroma) with OpenAI embeddings for fast similarity search.
- Use a conversational retrieval chain (LangChain) to allow follow-up questions and conversation memory.

Prerequisites
- Python 3.11+
- A working OpenAI API key (this project references the Cornell OpenAI gateway — use whatever key/endpoint your environment requires). Do NOT hard-code keys in files.

Quick start (Codespace or local)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Provide your OpenAI key securely (example, temporary for the session):

```bash
export OPENAI_API_KEY="sk-..."
```

3. Start the app (example command; replace with your app entrypoint):

```bash
# if your app is a Streamlit or Flask app, replace with the correct script name
streamlit run app.py
```

Usage
- Upload one or more files (PDF, TXT, or MD) by browsing or drag-and-drop in the UI.
- The chatbot will process files (this may take a few seconds) and indicate when it's ready.
- Ask questions in the chat box. Answers include citations (file name and page number when available).

Features
- Multiple file uploads and formats (PDFs are parsed page-by-page; text files are processed as a whole).
- Document chunking using RecursiveCharacterTextSplitter with overlap to preserve context.
- Vector embeddings created with Chroma using OpenAI's text-embedding-3-large model.
- ConversationalRetrievalChain (LangChain) provides conversational Q&A with memory support.
- Citations returned with answers (expandable list showing file and page).
- Fallback "stuffed-context" mode: if retrieval-augmented generation (RAG) can't find a relevant result, the system falls back to sending the text directly to the model.
- Scanned/image-based PDF guardrail: warns when PDFs appear scanned or unreadable.

Implementation notes
- Chunking: RecursiveCharacterTextSplitter (configurable chunk size and overlap).
- Embeddings: OpenAI text-embedding-3-large (changeable in configuration).
- Vector store: Chroma (persisted to a local directory by default).
- Retrieval: similarity search + conversational re-ranking via LangChain.

Security & secrets
- Never commit API keys or secrets to the repository. Use environment variables or GitHub Codespaces secrets.
- Quick checks (safe — does not print secrets):

```bash
if [ -n "${OPENAI_API_KEY+x}" ]; then echo "OPENAI_API_KEY is set"; else echo "OPENAI_API_KEY is NOT set"; fi
git grep -n "OPENAI_API_KEY" || true
```

If you discover an API key committed in the repo history, rotate/revoke it immediately and remove it from history (e.g., `git filter-repo` or BFG) — coordinate with course staff if needed.

Troubleshooting
- If embeddings or Chroma fail, check that the OpenAI key is valid and that required packages are installed.
- If PDFs are unreadable, verify OCR/scan detection and consider running OCR before ingesting.

Suggested next steps / customization
- Expose chunk size, embedding model, and vector store path in a config file or CLI flags.
- Add tests for chunking and retrieval to ensure reliable behavior across document types.

Questions or help
If you want, I can update this README in-place, add a `CONTRIBUTING.md`, or create a `check_secrets.sh` helper script to run the safe checks above. Tell me which and I'll implement it.



# Assignment 1 — Document Q&A Chatbot

A Streamlit app that ingests PDF, TXT, or MD files, creates vector embeddings, and provides a conversational chatbot with cited answers.

## Features and Design Justifications
Data ingestion
    - Supported files: .pdf parsed page by page with pypdf, .txt and .md decoded as whole files.
    - No OCR: scanned PDFs will show a warning and are ignored.
    - Metadata: PDFs carry source=filename and page=i, text files carry source=filename.

Chunking and indexing
    - Splitter: RecursiveCharacterTextSplitter with chunk_size=500, chunk_overlap=80.
    Rationale: 500 chars keep chunks semantically coherent while fitting model context and 80 overlap preserves cross-chunk continuity.
    - Embeddings: text-embedding-3-large via langchain_openai.
    - Vector store: in-memory Chroma.from_documents.
    Rationale: simple and involves zero setup with index resetting when files change.

Retrieval + chat
    - Retriever k: RETRIEVAL_K=12.
    Rationale: modest recall without flooding the model.
    - LLM: gpt-4o-mini via ChatOpenAI(temperature=0.2) for stable answers.
    - Conversation memory: ConversationBufferMemory to let follow-ups reuse chat history.
    - Citations: filenames and PDF page numbers are shown in a Sources expander.

Fallback behavior
    - If the retrieval chain returns empty or “I don’t know,” the app makes a second pass that stuffs only your file text into the prompt, capped at MAX_CONTEXT_CHARS=120000.
    - The fallback system message says: use only the provided document set. If the answer is not present, say it is not available in the provided documents. This prevents hallucinations. If nothing extractable is found, the app tells you so.

UI flow
    - Upload multiple files, then the app lists them and builds chunks, lastly vector index and chain are created.
    - Chat input is disabled until the index is ready.
    - Answers render with a spinner and an optional Sources panel.
    - Uploading new files invalidates the old index and rebuilds it automatically.

Error handling and guardrails
    - Warns on unreadable or empty files.
    - Catches vectorstore or chain build errors and reports them in the UI.
    - Will not answer from outside knowledge. If content is not in your files, it will say so to prevent hallucination.

Limitations
    - No OCR for scanned PDFs.
    - In-memory index only. Restart or file changes clear it.
    - Citations list sources, not highlighted spans.

## Prerequisites
- Python 3.11+
- Valid OpenAI API key (do NOT hard-code)

## Quick Start

1. Install dependencies
```bash
python3 -m pip install -r requirements.txt

2. Set API Key
```bash
export OPENAI_API_KEY="sk-..."
```

3. Run app
```bash
streamlit run chat_with_pdf.py
```

Open http://localhost:8501

## Usage
- Upload PDF/TXT/MD files via drag-and-drop or browse
- Wait for processing (a few seconds)
- Ask questions in chat — answers include citations

## Implementation
- Chunking: RecursiveCharacterTextSplitter (500 chars, 80 overlap)
- Embeddings: OpenAI text-embedding-3-large
- Vector store: Chroma (in-memory)
- Retrieval: Top-12 similarity search + conversational re-ranking

## Troubleshooting
- Import errors: python3 -m pip install -U -r requirements.txt
- Embedding fails: Check OPENAI_API_KEY and network access
- No text in PDF: Likely scanned — run OCR first
- Check environment:
```bash
python3 -c "import langchain; print(langchain.__version__)"
python3 -m pip list | grep -i langchain
if [ -n "${OPENAI_API_KEY+x}" ]; then echo "OPENAI_API_KEY is set"; else echo "NOT set"; fi
```




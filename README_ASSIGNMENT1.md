# Assignment 1 — Document Q&A Chatbot

A Streamlit app that ingests PDF, TXT, or MD files, creates vector embeddings, and provides a conversational chatbot with cited answers.

## Features
- Parse PDFs page-by-page; TXT/MD as whole files
- Split documents into overlapping chunks (RecursiveCharacterTextSplitter)
- Vector embeddings via OpenAI text-embedding-3-large stored in Chroma
- ConversationalRetrievalChain (LangChain) with memory for follow-ups
- Citations (filename + page) included with answers
- Fallback mode: stuffs full context when retrieval fails
- Warns on scanned/unreadable PDFs

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




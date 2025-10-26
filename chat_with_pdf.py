# import standard libraries
import os
import io
import streamlit as st
from pypdf import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain  components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# manual fallback to call OpenAI via Cornell endpoint
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url="https://api.ai.it.cornell.edu",
)

# Constants
MODEL_CHAT = "openai.gpt-4o-mini"
MODEL_EMBED = "openai.text-embedding-3-large"
RETRIEVAL_K = 12                             
CHUNK_SIZE = 500                              
CHUNK_OVERLAP = 80
MAX_CONTEXT_CHARS = 120_000                           

st.title("Chit-Chat with Your Documents 📄")

# Headers and instructions 
uploaded_files = st.file_uploader(
    "Upload your documents to get started",
    type=("txt", "pdf", "md"),
    accept_multiple_files=True
)

prompt_text = "Ask me about the documents you've uploaded" if uploaded_files else "Upload documents here"

# Chat history and make sure that messages show up in UI
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Before we get started, please upload at least one document, then give me a moment to read through it."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Parsing files

def _extract_text(file) -> str:
    """Return text for a single uploaded file (TXT/MD direct decode; PDF per-page extraction)."""
    name = file.name.lower()
    data = file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages.append(t)
        return "\n".join(pages).strip()

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


# Ingest and chunk documents

def make_documents(files) -> list[Document]:
    """Build Document objects.
    - TXT/MD: one Document per file (metadata: source=filename)
    - PDF: one Document per page (metadata: source=filename, page=i)
    """
    docs: list[Document] = []
    for f in files:
        fname = f.name
        if fname.lower().endswith(".pdf"):
            raw = f.getvalue()
            reader = PdfReader(io.BytesIO(raw))
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(
                        Document(page_content=text, metadata={"source": fname, "page": i})
                    )
        else:
            text = _extract_text(f).strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs


def chunk_docs(docs: list[Document],
               chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Split documents into smaller chunks; preserve metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def _files_key(files):
    """Lightweight fingerprint of uploaded files to detect changes."""
    return tuple(sorted((f.name, getattr(f, "size", None)) for f in files))


# Context for fallback 

def _build_context(files) -> str:
    """Concatenate full text of all files for the fallback stuffed-context call (capped)."""
    parts = []
    total = 0
    for f in files:
        text = _extract_text(f).strip()
        if not text:
            st.warning(f"No extractable text in {f.name} (possibly a scanned PDF).")
            continue
        header = f"### {f.name}\n"
        need = len(header) + len(text) + 2  # +2 for spacing
        # Stop if adding this file would exceed the cap
        if total + need > MAX_CONTEXT_CHARS:
            # add as much as fits
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > len(header):
                take = remaining - len(header)
                parts.append(header + text[:max(0, take)])
                total = MAX_CONTEXT_CHARS
            break
        parts.append(header + text)
        total += need
    return "\n\n".join(parts)


# Preview uploaded files and build index on change
if uploaded_files:
    with st.expander("These are the files you have uploaded", expanded=False):
        for f in uploaded_files:
            st.write("•", f.name)

    key = _files_key(uploaded_files)
    if st.session_state.get("files_key") != key:
        docs = make_documents(uploaded_files)
        chunks = chunk_docs(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        st.session_state["files_key"] = key
        st.session_state["docs"] = docs
        st.session_state["chunks"] = chunks
        # Reset downstream when files change
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("chain", None)

# Status line 
if "chunks" in st.session_state:
    print(f"Chunks ready")


# RAG components

def build_vectorstore(chunks) -> Chroma:
    """Create an in-memory Chroma vector store from chunks using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(model=MODEL_EMBED)
    return Chroma.from_documents(chunks, embedding=embeddings)


def build_chain(vectorstore) -> ConversationalRetrievalChain:
    """Create a conversational RAG chain (retriever + chat LLM + memory)."""
    llm = ChatOpenAI(model=MODEL_CHAT, temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        memory=memory,
        return_source_documents=True,
    )

# Vector store and chain after chunks are ready
if st.session_state.get("chunks") and "vectorstore" not in st.session_state:
    if not st.session_state["chunks"]:
        st.error("I couldn't find any readable text in your documents. Check to make sure your PDFs are not scanned images.")
    else:
        try:
            with st.status("I'm reading through your files...", expanded=True) as status:
                status.write("I'm analyzing your documents...")
                vs = build_vectorstore(st.session_state["chunks"])
                status.write("I'm setting up the connection between your documents and me...")
                st.session_state["vectorstore"] = vs
                st.session_state["chain"] = build_chain(vs)
                status.update(label="Ready! You can ask me questions about your files now.", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Something went wrong while I was getting things ready for you: {e}")


# Flow of chat

question = st.chat_input(
    prompt_text,
    disabled=not st.session_state.get("chain"),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if not st.session_state.get("chain"):
        st.error("Retriever not ready. Upload files to build the index before asking questions.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("I'm thinking..."):
                result = st.session_state["chain"].invoke({"question": question})
                answer = (result.get("answer", "") or "").strip()
                sources = result.get("source_documents", [])

                # RAG fallback if no answer
                rag_blank = answer.lower() in {"", "i don't know.", "i don't know", "idk"}
                if rag_blank:
                    context = _build_context(uploaded_files)
                    if not context.strip():
                        st.warning("No extractable text found in the uploaded files.")
                        answer = "I don't know."
                        st.markdown(answer)

                        q_short = (question or "")[:120].replace("\n", " ")
                        print(f"FALLBACK_USED question='{q_short}' context_chars=0 answer_len={len(answer)}")
                    else:
                        stream = client.chat.completions.create(
                            model=MODEL_CHAT,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a precise assistant. Use only the provided document set. "
                                        "If the answer is not present, say that you couldn't find the answer in the documents."
                                    ),
                                },
                                {"role": "system", "content": f"Document set:\n\n{context}"},
                                {"role": "user", "content": question},  # <-- only the current turn
                            ],
                            stream=True,
                        )
                        answer = st.write_stream(stream)
                        q_short = (question or "")[:120].replace("\n", " ")
                        ans_len = len(answer) if isinstance(answer, str) else -1
                        print(f"FALLBACK_USED question='{q_short}' context_chars={len(context)} answer_len={ans_len}")
                else:
                    st.markdown(answer)
                    if sources:
                        with st.expander("Here's where I found the information"):
                            for i, d in enumerate(sources, 1):
                                src = d.metadata.get("source", "(unknown)")
                                page = d.metadata.get("page")
                                st.write(f"[{i}] {src}" + (f", page {page}" if page else ""))
                    q_short = (question or "")[:120].replace("\n", " ")
                    src_count = len(sources) if sources else 0
                    print(f"RAG_USED question='{q_short}' sources={src_count} answer_len={len(answer)}")

    # Assistant turn in UI history
    st.session_state.messages.append({"role": "assistant", "content": answer})

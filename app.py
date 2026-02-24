import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_loader import load_uploaded_file, load_all_documents
from src.text_splitter import split_documents
from src.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    save_vectorstore,
    add_documents_to_vectorstore,
    delete_vectorstore,
    get_document_count
)
from src.qa_chain import get_answer, check_ollama_connection
from config import DATA_DIR, OLLAMA_BASE_URL, LLM_MODEL


# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to vectorstore."""
    all_documents = []

    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read()
        docs = load_uploaded_file(file_content, uploaded_file.name)
        all_documents.extend(docs)

    if not all_documents:
        return 0, "No valid documents found in uploaded files."

    # Split documents into chunks
    chunks = split_documents(all_documents)

    if not chunks:
        return 0, "No content could be extracted from the documents."

    # Create or update vectorstore
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = create_vectorstore(chunks)
    else:
        st.session_state.vectorstore = add_documents_to_vectorstore(
            st.session_state.vectorstore,
            chunks
        )

    # Save to disk
    save_vectorstore(st.session_state.vectorstore)

    return len(chunks), None


def process_directory_files():
    """Process files from the data directory."""
    documents = load_all_documents(DATA_DIR)

    if not documents:
        return 0, "No supported files found in the data directory."

    chunks = split_documents(documents)

    if not chunks:
        return 0, "No content could be extracted from the documents."

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = create_vectorstore(chunks)
    else:
        st.session_state.vectorstore = add_documents_to_vectorstore(
            st.session_state.vectorstore,
            chunks
        )

    save_vectorstore(st.session_state.vectorstore)

    return len(chunks), None


def clear_index():
    """Clear the vectorstore index."""
    delete_vectorstore()
    st.session_state.vectorstore = None
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    st.header("Document Management")

    # Ollama status
    ollama_ok, ollama_msg = check_ollama_connection()
    if ollama_ok:
        st.success(f"Ollama connected ({LLM_MODEL})")
    else:
        st.error("Ollama not connected")
        st.info(f"Start Ollama and run: ollama pull {LLM_MODEL}")

    st.divider()

    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose your files",
        type=["txt", "md", "xml", "pdf", "csv", "json", "xlsx", "xls", "docx", "html", "htm", "pptx"],
        accept_multiple_files=True,
        help="Supported: TXT, Markdown, XML, PDF, CSV, JSON, Excel (XLSX/XLS), Word (DOCX), HTML, PowerPoint (PPTX)"
    )

    if uploaded_files:
        if st.button("Process Uploaded Files", type="primary"):
            with st.spinner("Processing documents..."):
                count, error = process_uploaded_files(uploaded_files)
                if error:
                    st.error(error)
                else:
                    st.success(f"Processed {count} document chunks")

    st.divider()

    # Load from directory
    st.subheader("Load from Directory")
    st.caption(f"Path: `{DATA_DIR}`")

    if st.button("Load Directory Files"):
        with st.spinner("Loading documents from directory..."):
            count, error = process_directory_files()
            if error:
                st.error(error)
            else:
                st.success(f"Processed {count} document chunks")

    st.divider()

    # Index status
    st.subheader("Index Status")
    doc_count = get_document_count(st.session_state.vectorstore)
    st.metric("Indexed Chunks", doc_count)

    if doc_count > 0:
        if st.button("Clear Index", type="secondary"):
            clear_index()
            st.rerun()


# Main content
st.title("RAG Document Q&A For Madhura")
st.caption("Ask questions about your documents - answers are grounded only in the uploaded content")

# Check if ready to answer questions
if st.session_state.vectorstore is None or get_document_count(st.session_state.vectorstore) == 0:
    st.info("Upload and process documents using the sidebar to get started.")
else:
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What information are you looking for?",
        key="question_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear History", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

    # Process question
    if ask_button and question:
        with st.spinner("Searching documents and generating answer..."):
            result = get_answer(st.session_state.vectorstore, question)

            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "result": result
            })

    # Display chat history (most recent first)
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"**Q: {entry['question']}**")

            result = entry['result']

            if result.get("error"):
                st.error(result["error"])
            else:
                st.markdown(result["answer"])

                # Show sources in expander
                if result.get("sources"):
                    with st.expander("View Source Documents", expanded=False):
                        for j, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {j}:** `{source['source']}`")
                            st.markdown(f"*Relevance Score: {source['score']:.4f}*")
                            st.text(source['content'])
                            if j < len(result["sources"]):
                                st.divider()

            st.divider()


# Footer
st.markdown("---")
st.caption(f"Powered by Ollama ({LLM_MODEL}), HuggingFace Embeddings, and FAISS")

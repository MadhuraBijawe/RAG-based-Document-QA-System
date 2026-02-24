from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings
from config import VECTORSTORE_DIR


FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"


def create_vectorstore(documents: List[Document]) -> Optional[FAISS]:
    """Create a new FAISS vectorstore from documents."""
    if not documents:
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """Save vectorstore to disk."""
    vectorstore.save_local(str(FAISS_INDEX_PATH))


def load_vectorstore() -> Optional[FAISS]:
    """Load vectorstore from disk if it exists."""
    if not FAISS_INDEX_PATH.exists():
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


def add_documents_to_vectorstore(
    vectorstore: FAISS,
    documents: List[Document]
) -> FAISS:
    """Add new documents to existing vectorstore."""
    if not documents:
        return vectorstore

    vectorstore.add_documents(documents)
    return vectorstore


def delete_vectorstore() -> bool:
    """Delete the vectorstore from disk."""
    try:
        if FAISS_INDEX_PATH.exists():
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH)
        return True
    except Exception as e:
        print(f"Error deleting vectorstore: {e}")
        return False


def get_document_count(vectorstore: FAISS) -> int:
    """Get the number of documents in the vectorstore."""
    if vectorstore is None:
        return 0
    return vectorstore.index.ntotal

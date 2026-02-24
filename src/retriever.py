from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import TOP_K


def retrieve_documents(
    vectorstore: FAISS,
    query: str,
    top_k: int = TOP_K
) -> List[Document]:
    """Retrieve relevant documents for a query."""
    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search(query, k=top_k)
    return docs


def retrieve_documents_with_scores(
    vectorstore: FAISS,
    query: str,
    top_k: int = TOP_K
) -> List[Tuple[Document, float]]:
    """Retrieve relevant documents with similarity scores."""
    if vectorstore is None:
        return []

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    return docs_with_scores


def format_context(documents: List[Document]) -> str:
    """Format retrieved documents into context string."""
    if not documents:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Document {i} - Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)

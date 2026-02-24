from typing import Dict, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from src.retriever import retrieve_documents_with_scores, format_context
from config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


def check_ollama_connection() -> Tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if LLM_MODEL.split(":")[0] in model_names:
                return True, f"Connected to Ollama with {LLM_MODEL}"
            return False, f"Model {LLM_MODEL} not found. Run: ollama pull this {LLM_MODEL}"
        return False, "Ollama not responding"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Is it running?"
    except Exception as e:
        return False, f"Error: {str(e)}"


GROUNDED_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based ONLY on the provided context from documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer ONLY using information from the context above
- If the answer is not in the context, respond with: "I cannot find this information in the provided documents."
- When answering, cite the source document (e.g., "According to [Document 1]...")
- Be concise and accurate
- Do not make up information or use knowledge outside the provided context

Answer:"""
)


def get_llm() -> Optional[ChatOllama]:
    """Get configured Ollama LLM."""
    try:
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    except Exception:
        return None


def get_answer(
    vectorstore: FAISS,
    question: str
) -> Dict:
    """
    Get an answer to a question using RAG.

    Returns:
        Dict with keys: 'answer', 'sources', 'error'
    """
    result = {
        "answer": "",
        "sources": [],
        "error": None
    }

    # Check for Ollama connection
    llm = get_llm()
    if llm is None:
        result["error"] = "Could not connect to Ollama. Please ensure Ollama is running."
        return result

    # Check for vectorstore
    if vectorstore is None:
        result["error"] = "No documents have been indexed yet. Please upload and process documents first."
        return result

    try:
        # Retrieve relevant documents
        docs_with_scores = retrieve_documents_with_scores(vectorstore, question)

        if not docs_with_scores:
            result["answer"] = "I cannot find relevant information in the provided documents."
            return result

        # Extract documents and format context
        documents = [doc for doc, score in docs_with_scores]
        context = format_context(documents)

        # Format prompt
        prompt = GROUNDED_QA_PROMPT.format(
            context=context,
            question=question
        )

        # Get answer from LLM
        response = llm.invoke(prompt)
        result["answer"] = response.content

        # Add source information
        result["sources"] = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "score": float(score)
            }
            for doc, score in docs_with_scores
        ]

    except Exception as e:
        result["error"] = f"Error generating answer: {str(e)}"

    return result

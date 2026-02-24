# RAG Application - Production Documentation

## Overview
A full-stack RAG (Retrieval-Augmented Generation) application that ingests TXT and XML files, creates embeddings with FAISS, and provides a Streamlit web UI for document-grounded Q&A.

## Tech Stack
| Component | Technology |
|-----------|------------|
| LLM | Google Gemini (gemini-1.5-flash) |
| Embeddings | HuggingFace (sentence-transformers/all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Web UI | Streamlit |
| Backend | Python |
| Document Processing | LangChain |

## Project Structure
```
rag-app/
├── app.py                    # Streamlit main application
├── config.py                 # Configuration and environment variables
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment file
├── data/
│   └── documents/            # Folder for TXT and XML files
├── vectorstore/              # FAISS index storage
└── src/
    ├── __init__.py
    ├── document_loader.py    # TXT and XML file ingestion
    ├── text_splitter.py      # Document chunking
    ├── embeddings.py         # HuggingFace embeddings wrapper
    ├── vectorstore.py        # FAISS operations
    ├── retriever.py          # Document retrieval logic
    └── qa_chain.py           # OLLAMA QA chain with grounding
```

## Quick Start

### 1. Install Dependencies
```bash
cd rag-app
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Usage
1. Upload TXT or XML files using the sidebar
2. Click "Process Documents" to index them
3. Ask questions in the main input area
4. View answers with source citations

## Architecture

### Document Processing Pipeline
1. **Ingestion**: Load TXT/XML files from uploads or `data/documents/`
2. **Chunking**: Split documents into 1000-char chunks with 200-char overlap
3. **Embedding**: Generate vectors using HuggingFace `all-MiniLM-L6-v2`
4. **Storage**: Store vectors in FAISS index (persisted to `vectorstore/`)

### Query Pipeline
1. **Embedding**: Convert user question to vector
2. **Retrieval**: Find top-5 similar document chunks via FAISS
3. **Generation**: Send context + question to Gemini with grounding prompt
4. **Response**: Return answer with source citations

### Grounding Strategy
The system uses a carefully crafted prompt to ensure answers are grounded:
- Model instructed to ONLY use provided context
- Must cite source documents
- Returns "I cannot find this information" when answer not in documents
- Prevents hallucination by design

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| GOOGLE_API_KEY | Required | Your Google AI API key |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | HuggingFace embedding model |
| CHUNK_SIZE | 1000 | Characters per document chunk |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| TOP_K | 5 | Number of chunks to retrieve |

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Environment Variables for Production
- Set `GOOGLE_API_KEY` as a secret
- Consider using persistent volume for `vectorstore/`
- Enable Streamlit authentication for public deployments

## Error Handling
- Invalid file formats: User-friendly error message
- Empty documents: Skipped with warning
- API errors: Retry with exponential backoff
- Missing API key: Clear setup instructions displayed

## Security Considerations
- API keys stored in environment variables only
- File uploads validated for type and size
- No external network access except to Gemini API
- Document content never logged or exposed

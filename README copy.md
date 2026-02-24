# RAG Document Q&A

A **Retrieval-Augmented Generation (RAG)** application that lets you ask questions about your documents. The app retrieves relevant content from your uploaded documents and generates accurate, grounded answers using a local LLM.

## What is RAG?

RAG combines two powerful techniques:
1. **Retrieval**: Find relevant chunks from your documents using semantic search
2. **Generation**: Use an LLM to generate answers based only on the retrieved content

This ensures answers are **grounded in your documents** - the AI won't make things up!

## Features

- Upload TXT and XML documents
- Automatic text chunking and indexing
- Semantic search using FAISS vector database
- Local LLM inference via Ollama (no API costs, runs on your machine)
- Source citation with relevance scores
- Chat history within session
- Persistent vector index (survives app restarts)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Document   │───>│    Text      │───>│    FAISS         │   │
│  │   Loader     │    │   Splitter   │    │   Vectorstore    │   │
│  │  (TXT/XML)   │    │  (Chunking)  │    │   (Embeddings)   │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                   │             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────▼─────────┐   │
│  │   Answer     │<───│   Ollama     │<───│    Retriever     │   │
│  │   Display    │    │   LLM        │    │   (Top-K docs)   │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-app/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (Ollama settings)
├── data/
│   └── documents/         # Place documents here for batch loading
├── vectorstore/           # Persisted FAISS index
└── src/
    ├── document_loader.py # Load TXT and XML files
    ├── text_splitter.py   # Split documents into chunks
    ├── embeddings.py      # HuggingFace embeddings model
    ├── vectorstore.py     # FAISS vector database operations
    ├── retriever.py       # Semantic search and retrieval
    └── qa_chain.py        # Ollama LLM and answer generation
```

## Prerequisites

- **Python 3.10+**
- **Ollama** - Local LLM runtime ([Download here](https://ollama.ai/download))

## Installation

### Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd rag-app
```

### Step 2: Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and setup Ollama

1. **Download Ollama** from [ollama.ai/download](https://ollama.ai/download) and install it

2. **Start Ollama** (open a new terminal and keep it running):
   ```bash
   ollama serve
   ```

3. **Pull the LLM model** (in another terminal):
   ```bash
   ollama pull llama3.2
   ```

   This downloads the `llama3.2` model (~2GB). Other options:
   - `ollama pull mistral` - Good general purpose model
   - `ollama pull phi3` - Smaller, faster model
   - `ollama pull gemma2` - Google's open model

## Configuration

Edit the `.env` file to customize:

```env
# Ollama server URL (default is localhost)
OLLAMA_BASE_URL=http://localhost:11434

# Model to use (must be pulled with ollama pull <model>)
OLLAMA_MODEL=llama3.2
```

## Running the Application

### Step 1: Make sure Ollama is running

In one terminal:
```bash
ollama serve
```

### Step 2: Start the Streamlit app

In another terminal (with venv activated):
```bash
streamlit run app.py
```

### Step 3: Open in browser

The app will open automatically at: **http://localhost:8501**

## How to Use

### 1. Upload Documents

- Click **"Browse files"** in the sidebar
- Select TXT or XML files
- Click **"Process Uploaded Files"**

### 2. Or Load from Directory

- Place files in `data/documents/` folder
- Click **"Load Directory Files"** in the sidebar

### 3. Ask Questions

- Type your question in the input box
- Click **"Ask"**
- View the answer with source citations

### 4. View Sources

- Expand **"View Source Documents"** to see which document chunks were used
- Each source shows relevance score (lower = more relevant)

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | Ollama (llama3.2) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| Framework | LangChain |

## Troubleshooting

### "Ollama not connected" error

1. Make sure Ollama is running: `ollama serve`
2. Check if the model is downloaded: `ollama list`
3. If model not listed, pull it: `ollama pull llama3.2`

### "Cannot connect to Ollama" error

- Verify Ollama is running on port 11434
- Check `.env` has correct `OLLAMA_BASE_URL`

### Slow first response

- First query downloads the embedding model (~90MB)
- Subsequent queries will be faster

### Out of memory

- Try a smaller model: `ollama pull phi3`
- Update `.env`: `OLLAMA_MODEL=phi3`

## Configuration Options

Edit `config.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `LLM_TEMPERATURE` | 0.1 | Response creativity (0-1) |
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval configuration
TOP_K = 5

# LLM configuration (Ollama model)
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TEMPERATURE = 0.1

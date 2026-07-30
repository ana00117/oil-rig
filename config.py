from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "BAAI/bge-reranker-base"

TEMPERATURE = 0.1
MAX_TOKENS = 1024

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

TOP_K_RETRIEVAL = 8
FINAL_TOP_K = 4

VECTOR_DB_PATH = "vector_store"

UPLOAD_FOLDER = "uploads"

ALLOWED_FILE_TYPES = [".pdf"]

MEMORY_SIZE = 10

APP_TITLE = "ChemE RAG Assistant"
PAGE_ICON = ""


.strip()

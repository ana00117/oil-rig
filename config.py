import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.1-8b-instant"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 8

UPLOAD_FOLDER = "uploads"
VECTOR_STORE_PATH = "faiss_index"

SUPPORTED_FILE_TYPES = ["pdf"]

PAGE_TITLE = "ChemE RAG Assistant"
PAGE_ICON = "🤖"

MAX_HISTORY = 10

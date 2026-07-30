import os
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"

TEMPERATURE = 0.1
MAX_TOKENS = 1024

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]

TOP_K_RETRIEVAL = 8

FINAL_TOP_K = 4
VECTOR_DB_PATH = "vector_store"
UPLOAD_FOLDER = "uploads"

ALLOWED_FILE_TYPES = [".pdf"]
APP_TITLE = "Intelligent RAG Chatbot"

SYSTEM_PROMPT = """
You are an expert document assistant.

Rules:

1. Answer ONLY using the supplied context.

2. Never invent information.

3. If the answer cannot be found,
respond exactly:

'I could not find this information in the uploaded documents.'

4. Be concise but complete.

5. If multiple retrieved passages support the answer,
combine them naturally.

6. Mention document names and page numbers whenever available.

7. Never mention information outside the provided context.

8. Use bullet points whenever appropriate.
"""

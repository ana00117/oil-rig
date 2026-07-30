"""End-to-end RAG pipeline: process documents, then answer questions."""

import os
from dotenv import load_dotenv

from document_loader import load_pdfs_from_uploads
from vector_store import (
    chunk_documents,
    build_vector_store,
    save_vector_store,
    retrieve,
    get_embedding_model,
)
from prompt import build_prompt

load_dotenv()


class RAGPipeline:
    def __init__(self, model_name="llama-3.3-70b-versatile", top_k=4):
        self.top_k = top_k
        self.model_name = model_name
        self.embedding_model = get_embedding_model()
        self.vector_store = None
        self.llm = self._load_llm()

    def _load_llm(self):
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

        return ChatGroq(
            model=self.model_name, groq_api_key=api_key, temperature=0
        )

    def process_documents(self, uploaded_files):
        """Extract, chunk, embed, and index uploaded PDFs. Returns (num_pages, num_chunks)."""
        documents = load_pdfs_from_uploads(uploaded_files)
        if not documents:
            raise ValueError("No extractable text found in the uploaded PDFs.")

        chunks = chunk_documents(documents)
        self.vector_store = build_vector_store(chunks, self.embedding_model)
        save_vector_store(self.vector_store)
        return len(documents), len(chunks)

    def ask(self, question):
        """Retrieve relevant chunks and generate a grounded answer with sources."""
        if self.vector_store is None:
            raise ValueError("No documents processed yet. Upload and process PDFs first.")

        chunks = retrieve(self.vector_store, question, k=self.top_k)
        prompt_text = build_prompt(chunks, question)
        response = self.llm.invoke(prompt_text)

        sources = [
            {"source": c.metadata.get("source"), "page": c.metadata.get("page")}
            for c in chunks
        ]
        return response.content, sources

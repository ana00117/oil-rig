"""Chunking, embeddings, and FAISS vector store."""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split extracted page text into overlapping chunks.

    Args:
        documents: list of {"text", "source", "page"} dicts.

    Returns:
        List of langchain Document objects with source/page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for chunk_text in splitter.split_text(doc["text"]):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={"source": doc["source"], "page": doc["page"]},
                )
            )
    return chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks, embedding_model=None):
    """Embed chunks and build a FAISS index."""
    embedding_model = embedding_model or get_embedding_model()
    return FAISS.from_documents(chunks, embedding_model)


def save_vector_store(vector_store, path="vector_store/saved_index"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path="vector_store/saved_index", embedding_model=None):
    embedding_model = embedding_model or get_embedding_model()
    return FAISS.load_local(
        path, embedding_model, allow_dangerous_deserialization=True
    )


def retrieve(vector_store, query, k=4):
    """Return the top-k most relevant chunks for a query."""
    return vector_store.similarity_search(query, k=k)
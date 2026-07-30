import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from config import (
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
)


class VectorStoreManager:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = None

    def create_vector_store(
        self,
        documents: List[Document],
    ):
        """
        Build a FAISS index from documents.
        """

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
        )

        return self.vector_store

    def save(self):
        """
        Save FAISS index locally.
        """

        if self.vector_store is None:
            raise ValueError("Vector store has not been created.")

        self.vector_store.save_local(VECTOR_DB_PATH)

    def load(self):
        """
        Load saved FAISS index.
        """

        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(
                f"{VECTOR_DB_PATH} does not exist."
            )

        self.vector_store = FAISS.load_local(
            VECTOR_DB_PATH,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 8,
    ):
        """
        Standard similarity search.
        """

        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")

        return self.vector_store.similarity_search(
            query=query,
            k=k,
        )


    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 8,
    ):
        """
        Returns documents with similarity scores.
        """

        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

    
    def as_retriever(
        self,
        k: int = 8,
    ):
        """
        Convert to LangChain retriever.
        """

        if self.vector_store is None:
            raise ValueError("Vector store is not loaded.")

        return self.vector_store.as_retriever(
            search_kwargs={
                "k": k
            }
        )

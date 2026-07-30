from typing import List

from langchain.schema import Document
from rank_bm25 import BM25Okapi

from config import TOP_K_RETRIEVAL
from vector_store import VectorStoreManager


class HybridRetriever:

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        documents: List[Document],
    ):

        self.vector_store = vector_store_manager

        self.documents = documents

   
        self.tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    

    def bm25_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
    ) -> List[Document]:

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(scores, self.documents),
            key=lambda x: x[0],
            reverse=True,
        )

        return [doc for _, doc in ranked[:k]]

  

    def semantic_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
    ) -> List[Document]:

        return self.vector_store.similarity_search(
            query=query,
            k=k,
        )


    def hybrid_search(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
    ) -> List[Document]:

        semantic_docs = self.semantic_search(query, k)

        bm25_docs = self.bm25_search(query, k)

        merged = []
        seen = set()

       
        for doc in semantic_docs + bm25_docs:

            doc_id = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.metadata.get("chunk_id"),
            )

            if doc_id not in seen:

                seen.add(doc_id)

                merged.append(doc)

        return merged[:k]

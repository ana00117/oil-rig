import pickle
from typing import List

import faiss
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    MODEL_NAME,
    TOP_K,
    VECTOR_STORE_PATH,
)

from prompts import build_prompt
from utils import Chunk, format_context


class RAG:

    def __init__(self):

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.index = None

        self.chunks = []

        self.bm25 = None

        self.history = []

    def build(self, chunks: List[Chunk]):

        self.chunks = chunks

        embeddings = self.embedding_model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings.astype("float32")
        )

        corpus = [
            chunk.text.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(corpus)

        faiss.write_index(
            self.index,
            f"{VECTOR_STORE_PATH}/index.faiss",
        )

        with open(
            f"{VECTOR_STORE_PATH}/chunks.pkl",
            "wb",
        ) as f:

            pickle.dump(
                chunks,
                f,
            )

    def load(self):

        self.index = faiss.read_index(
            f"{VECTOR_STORE_PATH}/index.faiss"
        )

        with open(
            f"{VECTOR_STORE_PATH}/chunks.pkl",
            "rb",
        ) as f:

            self.chunks = pickle.load(f)

        corpus = [
            chunk.text.lower().split()
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        question: str,
    ):

        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        _, semantic_indices = self.index.search(
            np.array(
                [query_embedding],
                dtype="float32",
            ),
            TOP_K,
        )

        semantic = [
            self.chunks[i]
            for i in semantic_indices[0]
        ]

        keyword_scores = self.bm25.get_scores(
            question.lower().split()
        )

        keyword_indices = np.argsort(
            keyword_scores
        )[::-1][:TOP_K]

        keyword = [
            self.chunks[i]
            for i in keyword_indices
        ]

        merged = []

        seen = set()

        for chunk in semantic + keyword:

            key = (
                chunk.source,
                chunk.page,
                chunk.text[:100],
            )

            if key not in seen:

                seen.add(key)

                merged.append(chunk)

        return merged[:TOP_K]

    def ask(
        self,
        question: str,
    ):

        chunks = self.retrieve(
            question
        )

        context = format_context(
            chunks
        )

        history = "\n".join(
            self.history[-10:]
        )

        prompt = build_prompt(
            question,
            context,
            history,
        )

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response.choices[0].message.content

        self.history.append(
            f"User: {question}"
        )

        self.history.append(
            f"Assistant: {answer}"
        )

        sources = [
            {
                "document": chunk.source,
                "page": chunk.page,
            }
            for chunk in chunks
        ]

        return answer, sources

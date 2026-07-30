import os
import json
import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    VECTOR_STORE_PATH,
    TOP_K,
)


class VectorStore:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.index = None

        self.documents = []

    def build(self, documents):

        self.documents = documents

        texts = [
            doc["text"]
            for doc in documents
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings.astype("float32")
        )

        self.save()

    def save(self):

        os.makedirs(
            VECTOR_STORE_PATH,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            os.path.join(
                VECTOR_STORE_PATH,
                "index.faiss",
            ),
        )

        with open(
            os.path.join(
                VECTOR_STORE_PATH,
                "metadata.pkl",
            ),
            "wb",
        ) as f:

            pickle.dump(
                self.documents,
                f,
            )

        with open(
            os.path.join(
                VECTOR_STORE_PATH,
                "config.json",
            ),
            "w",
        ) as f:

            json.dump(
                {
                    "embedding_model": EMBEDDING_MODEL,
                },
                f,
                indent=4,
            )

    def load(self):

        self.index = faiss.read_index(
            os.path.join(
                VECTOR_STORE_PATH,
                "index.faiss",
            )
        )

        with open(
            os.path.join(
                VECTOR_STORE_PATH,
                "metadata.pkl",
            ),
            "rb",
        ) as f:

            self.documents = pickle.load(f)

    def exists(self):

        return os.path.exists(
            os.path.join(
                VECTOR_STORE_PATH,
                "index.faiss",
            )
        )

    def search(
        self,
        query,
        top_k=TOP_K,
    ):

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores, indices = self.index.search(
            np.array(
                [embedding],
                dtype="float32",
            ),
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:
                continue

            doc = self.documents[index].copy()

            doc["score"] = float(score)

            results.append(doc)

        return results

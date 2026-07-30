from typing import List

import torch
from langchain.schema import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import FINAL_TOP_K, RERANKER_MODEL


class Reranker:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL
        )
        self.model.eval()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = FINAL_TOP_K,
    ) -> List[Document]:

        if not documents:
            return []

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        with torch.no_grad():
            scores = (
                self.model(**inputs)
                .logits.squeeze(-1)
                .cpu()
                .tolist()
            )

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            document
            for _, document in ranked[:top_k]
        ]

from typing import List

from langchain.schema import Document
from langchain_groq import ChatGroq

from config import (
    FINAL_TOP_K,
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)

from memory import ConversationMemory
from prompts import prompt
from reranker import Reranker
from retriever import HybridRetriever


class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRetriever,
        memory: ConversationMemory,
    ):
        self.retriever = retriever
        self.memory = memory
        self.reranker = Reranker()

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

    def build_context(
        self,
        documents: List[Document],
    ) -> str:

        context = []

        for document in documents:

            source = document.metadata.get("source", "Unknown")

            page = document.metadata.get("page", "Unknown")

            context.append(
                f"""
Document: {source}
Page: {page}

{document.page_content}
"""
            )

        return "\n\n".join(context)

    def ask(
        self,
        question: str,
    ):

        retrieved_documents = self.retriever.hybrid_search(
            query=question
        )

        reranked_documents = self.reranker.rerank(
            query=question,
            documents=retrieved_documents,
            top_k=FINAL_TOP_K,
        )

        context = self.build_context(reranked_documents)

        history = self.memory.get_context()

        chain = prompt | self.llm

        response = chain.invoke(
            {
                "history": history,
                "context": context,
                "question": question,
            }
        )

        answer = response.content

        self.memory.add_user_message(question)
        self.memory.add_assistant_message(answer)

        sources = []

        seen = set()

        for document in reranked_documents:

            source = (
                document.metadata.get("source"),
                document.metadata.get("page"),
            )

            if source not in seen:

                seen.add(source)

                sources.append(
                    {
                        "document": source[0],
                        "page": source[1],
                    }
                )

        return {
            "answer": answer,
            "sources": sources,
        }

    def clear_memory(self):
        self.memory.clear()

from groq import Groq

from config import (
    GROQ_API_KEY,
    MODEL_NAME,
)

from prompt import build_prompt

from vector_store import VectorStore


class RAGPipeline:

    def __init__(self):

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.vector_store = VectorStore()

        if self.vector_store.exists():

            self.vector_store.load()

        self.chat_history = []

    def build_vector_store(self, documents):

        self.vector_store.build(documents)

    def ask(self, question):

        retrieved_docs = self.vector_store.search(question)

        context = ""

        for doc in retrieved_docs:

            context += (
                f"Document: {doc['source']}\n"
                f"Page: {doc['page']}\n\n"
                f"{doc['text']}\n\n"
            )

        history = ""

        if self.chat_history:

            history = "\n".join(
                self.chat_history[-10:]
            )

        prompt = build_prompt(
            question=question,
            context=context,
            history=history,
        )

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        self.chat_history.append(
            f"User: {question}"
        )

        self.chat_history.append(
            f"Assistant: {answer}"
        )

        sources = []

        seen = set()

        for doc in retrieved_docs:

            key = (
                doc["source"],
                doc["page"],
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "document": doc["source"],
                    "page": doc["page"],
                    "text": doc["text"][:250] + "...",
                }
            )

        return answer, sources

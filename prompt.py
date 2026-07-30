"""Prompt template and guardrail for grounded answer generation (oil rig domain)."""

SYSTEM_PROMPT = """You are an assistant for oil rig operations documentation
(drilling reports, equipment manuals, maintenance logs, inspection records,
safety procedures, and incident reports).

Answer only from the supplied context. If the answer is not available, say:
"I could not find this information in the uploaded documents."

Do not invent facts, specifications, or safety procedures. Mention the source
document and page number when available. For any safety-critical or
operational information, remind the user to verify against the original
document before acting on it."""


def build_prompt(context_chunks, question):
    """
    Build the final prompt sent to the LLM.

    Args:
        context_chunks: list of langchain Document objects (retrieved chunks).
        question: the user's question.

    Returns:
        A single prompt string combining context and question.
    """
    context_blocks = []
    for chunk in context_chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        context_blocks.append(f"[{source}, page {page}]\n{chunk.page_content}")

    context_text = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

Context:
{context_text}

Question: {question}

Answer:"""

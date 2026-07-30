import os
import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdfs(pdf_paths: List[str]) -> List[Chunk]:
    chunks = []

    for pdf_path in pdf_paths:

        reader = PdfReader(pdf_path)

        filename = os.path.basename(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if not text:
                continue

            text = clean_text(text)

            chunks.append(
                Chunk(
                    text=text,
                    source=filename,
                    page=page_number,
                )
            )

    return chunks


def split_chunks(chunks: List[Chunk]) -> List[Chunk]:

    split_documents = []

    for chunk in chunks:

        text = chunk.text

        start = 0

        while start < len(text):

            end = min(start + CHUNK_SIZE, len(text))

            split_documents.append(
                Chunk(
                    text=text[start:end],
                    source=chunk.source,
                    page=chunk.page,
                )
            )

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return split_documents


def format_context(chunks: List[Chunk]) -> str:

    context = []

    for chunk in chunks:

        context.append(
            f"""Document: {chunk.source}
Page: {chunk.page}

{chunk.text}"""
        )

    return "\n\n".join(context)

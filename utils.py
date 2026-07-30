import os
import re
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    documents = []

    for pdf_path in pdf_paths:

        reader = PdfReader(pdf_path)

        filename = os.path.basename(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if not text:
                continue

            text = clean_text(text)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_number,
                    },
                )
            )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index

    return chunks


def format_context(documents: List[Document]) -> str:

    context = []

    for document in documents:

        context.append(
            f"""
Document: {document.metadata['source']}
Page: {document.metadata['page']}

{document.page_content}
"""
        )

    return "\n\n".join(context)

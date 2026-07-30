from pathlib import Path
import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
)


class DocumentLoader:
  

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )



    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted PDF text.
        """

        if not text:
            return ""

       
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.replace("\x00", "")

        return text.strip()


    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Read one PDF and return page Documents.
        """

        documents = []

        reader = PdfReader(pdf_path)

        filename = Path(pdf_path).name

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            text = self.clean_text(text)

            if not text:
                continue

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

    def load_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """
        Read multiple PDFs.
        """

        all_docs = []

        for pdf in pdf_paths:

            try:
                docs = self.load_pdf(pdf)
                all_docs.extend(docs)

            except Exception as e:

                print(f"Error loading {pdf}: {e}")

        return all_docs

 
    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Chunk the documents while preserving metadata.
        """

        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = i

        return chunks


    def load_and_split(
        self,
        pdf_paths: List[str],
    ) -> List[Document]:
        """
        Complete pipeline.
        """

        docs = self.load_multiple_pdfs(pdf_paths)

        chunks = self.split_documents(docs)

        return chunks

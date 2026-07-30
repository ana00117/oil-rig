import os
import re

from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:

    def __init__(self):
        self.documents = []

    def clean_text(self, text):

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def load_pdfs(self, pdf_paths):

        self.documents = []

        for pdf_path in pdf_paths:

            reader = PdfReader(pdf_path)

            filename = os.path.basename(pdf_path)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if not text:
                    continue

                text = self.clean_text(text)

                chunks = self.chunk_text(text)

                for chunk in chunks:

                    self.documents.append(
                        {
                            "text": chunk,
                            "source": filename,
                            "page": page_number,
                        }
                    )

        return self.documents

    def chunk_text(self, text):

        chunks = []

        step = CHUNK_SIZE - CHUNK_OVERLAP

        start = 0

        while start < len(text):

            end = min(start + CHUNK_SIZE, len(text))

            if end < len(text):

                while end > start and text[end] != " ":
                    end -= 1

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(chunk)

            start += step

        return chunks

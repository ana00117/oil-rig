"""PDF loading and text extraction with page-level metadata."""

from pathlib import Path
from pypdf import PdfReader


def load_pdfs(file_paths):
    """
    Extract text from one or more PDF files, skipping empty pages.

    Args:
        file_paths: list of paths (str or Path) to PDF files.

    Returns:
        List of dicts: {"text": str, "source": str, "page": int}
    """
    documents = []

    for file_path in file_paths:
        file_path = Path(file_path)
        reader = PdfReader(str(file_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            documents.append({
                "text": text.strip(),
                "source": file_path.name,
                "page": page_num,
            })

    return documents


def load_pdfs_from_uploads(uploaded_files, save_dir="documents"):
    """
    Save Streamlit UploadedFile objects to disk, then extract text.

    Args:
        uploaded_files: list of Streamlit UploadedFile objects.
        save_dir: directory to save the uploaded PDFs.

    Returns:
        List of dicts: {"text": str, "source": str, "page": int}
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        dest = save_path / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(dest)

    return load_pdfs(saved_paths)

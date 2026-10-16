"""Load PDF / Markdown / text files into LangChain Document objects."""
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .cleaning import clean_pages

SUPPORTED = {".pdf", ".md", ".markdown", ".txt"}


def load_path(path: Path, display_name: str | None = None) -> list[Document]:
    """Load one file. PDFs give one Document per page; text files give one Document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()

        # Remove repeated headers/footers so we don't waste tokens on them
        cleaned = clean_pages([d.page_content for d in docs])
        for d, text in zip(docs, cleaned):
            d.page_content = text

        # Drop pages that are empty after cleaning
        docs = [d for d in docs if d.page_content.strip()]
    elif suffix in SUPPORTED:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Use a friendly file name as the "source" so we can cite it later.
    for d in docs:
        d.metadata["source"] = display_name or path.name
    return docs


def load_folder(folder: str) -> list[Document]:
    """Load every supported file in a folder."""
    docs: list[Document] = []
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in SUPPORTED:
            docs.extend(load_path(p))
    return docs


def load_uploaded(uploaded_file) -> list[Document]:
    """Load a Streamlit UploadedFile by saving it to a temp file first."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        return load_path(tmp_path, display_name=uploaded_file.name)
    finally:
        tmp_path.unlink(missing_ok=True)

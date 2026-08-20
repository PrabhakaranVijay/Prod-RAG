import os
import tempfile
from time import perf_counter
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
)

from dotenv import load_dotenv

from app.core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)


def _log_loaded(documents, start: float):
    logger.info(
        "Document loading completed",
        extra={
            "latency_ms": int((perf_counter() - start) * 1000),
            "input_tokens": None,
            "output_tokens": None,
            "document_count": len(documents),
        },
    )
    return documents


def load_text_file(file_path: str):
    start = perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(file_path.encode())
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        return _log_loaded(documents, start)
    finally:
        os.remove(temp_file_path)  # Clean up the temporary file


def load_web_page(url: str):
    start = perf_counter()
    loader = WebBaseLoader(url)
    documents = loader.load()
    return _log_loaded(documents, start)


def load_pdf(file_path: str):
    start = perf_counter()
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return _log_loaded(documents, start)

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text file content for testing."
    load_text_file(sample_text)

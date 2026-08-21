from pathlib import Path
from time import perf_counter
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)

from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentParseError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    
    @staticmethod
    def _log_and_return(documents: List[Document], start: float) -> List[Document]:
        """Helper to avoid duplicating log boilerplate across loaders."""
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

    @classmethod
    def load(cls, source: str) -> List[Document]:
        """
        Universal loader that automatically routes files by extension 
        or handles web URLs.
        """
        start = perf_counter()

        # Handle Web URLs directly
        if source.startswith(("http://", "https://")):
            try:
                loader = WebBaseLoader(source)
                documents = loader.load()
                return cls._log_and_return(documents, start)
            except Exception as error:
                logger.exception("Failed to load web document", extra={"source": source})
                raise DocumentParseError(f"Failed to fetch or parse web URL '{source}': {error}") from error

        # Handle local files based on suffix
        path = Path(source)
        if not path.exists():
            raise DocumentNotFoundError(f"Local document file not found: {source}")

        suffix = path.suffix.lower()

        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
        }

        if suffix not in loaders:
            raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'. Supported extensions: .pdf, .txt, .csv")

        try:
            loader = loaders[suffix](str(path))
            documents = loader.load()
            return cls._log_and_return(documents, start)
        except Exception as error:
            logger.exception("Failed to parse local document", extra={"source": source})
            raise DocumentParseError(f"Failed to parse document '{source}': {error}") from error


import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
)

logger = logging.getLogger(__name__)

class DocumentLoader:
    
    @staticmethod
    def _log_and_return(documents: List[Document]) -> List[Document]:
        """Helper to avoid duplicating log boilerplate across loaders."""
        for doc in documents:
            logger.info(f"Loaded document: {doc.page_content[:100]}...")
            logger.info(f"Metadata: {doc.metadata}")
        return documents

    @classmethod
    def load(cls, source: str) -> List[Document]:
        """
        Universal loader that automatically routes files by extension 
        or handles web URLs.
        """
        # Handle Web URLs directly
        if source.startswith(("http://", "https://")):
            loader = WebBaseLoader(source)
            return cls._log_and_return(loader.load())

        # Handle local files based on suffix
        suffix = Path(source).suffix.lower()

        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
        }

        if suffix not in loaders:
            raise ValueError(f"Unsupported file type: {suffix}")

        loader = loaders[suffix](source)
        return cls._log_and_return(loader.load())
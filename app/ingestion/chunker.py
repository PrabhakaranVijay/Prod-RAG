from time import perf_counter
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger

logger = get_logger(__name__)

class DocumentChunker:

    @staticmethod
    def split(
        documents: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Splits documents into smaller chunks using RecursiveCharacterTextSplitter.
        """
        start = perf_counter()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )
        
        chunks = text_splitter.split_documents(documents)
        
        logger.info(
            "Document chunking completed",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
                "document_count": len(documents),
                "chunk_count": len(chunks),
            },
        )
        return chunks

    def recursive_split(
        self, 
        documents: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,        
    ) -> List[Document]:
        """
        Recursively splits documents into smaller chunks.
        """
        start = perf_counter()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, 
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(
            "Document chunking completed",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
                "document_count": len(documents),
                "chunk_count": len(chunks),
            },
        )
        return chunks

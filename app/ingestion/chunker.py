import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )
        
        chunks = text_splitter.split_documents(documents)
        
        logger.info(
            f"Split {len(documents)} source documents into {len(chunks)} chunks "
            f"(size={chunk_size}, overlap={chunk_overlap})."
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
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, 
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )
        logger.info(
            f"Recursively splitting {len(documents)} documents into chunks "
            f"(size={chunk_size}, overlap={chunk_overlap})."
        )
        return splitter.split_documents(documents)
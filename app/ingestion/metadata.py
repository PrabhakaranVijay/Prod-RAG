from typing import List, Dict, Any

from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)

class MetadataEnricher:

    @staticmethod
    def enrich(documents: List[Document], extra_metadata: Dict[str, Any]) -> List[Document]:
        """
        Enriches a list of documents with additional metadata (e.g., source, author, category).
        """
        for doc in documents:
            doc.metadata.update(extra_metadata)
            
        logger.info(
            "Document metadata enrichment completed",
            extra={
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "document_count": len(documents),
            },
        )
        return documents

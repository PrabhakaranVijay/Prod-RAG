import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MetadataEnricher:

    @staticmethod
    def enrich(documents: List[Document], extra_metadata: Dict[str, Any]) -> List[Document]:
        """
        Enriches a list of documents with additional metadata (e.g., source, author, category).
        """
        for doc in documents:
            doc.metadata.update(extra_metadata)
            
        logger.info(f"Enriched {len(documents)} documents with metadata: {list(extra_metadata.keys())}")
        return documents
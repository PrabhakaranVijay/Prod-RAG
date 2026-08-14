import logging

from app.ingestion.chunker import DocumentChunker
from app.ingestion.loader import DocumentLoader
from app.ingestion.metadata import MetadataEnricher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionPipeline:

    @staticmethod
    def run(source: str, metadata: dict, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Executes the full ingestion pipeline: Load -> Enrich Metadata -> Chunk.
        """
        logger.info(f"Starting ingestion pipeline for source: {source}")

        # 1. Load documents
        raw_documents = DocumentLoader.load(source)

        # 2. Add metadata
        enriched_documents = MetadataEnricher.enrich(raw_documents, metadata)

        # 3. Chunk documents
        document_chunks = DocumentChunker.split(
            enriched_documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

        logger.info("Ingestion pipeline completed successfully.")
        return document_chunks

if __name__ == "__main__":
    # Example usage:
    chunks = IngestionPipeline.run(
        source="data/raw/company_handbook.txt",
        metadata={"category": "documentation", "author": "admin"},
        chunk_size=500,
        chunk_overlap=50,
    )
    print(f"Total chunks ready for vector store: {len(chunks)}")
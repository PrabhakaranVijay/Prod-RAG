from time import perf_counter

from app.core.logging import get_logger
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loader import DocumentLoader
from app.ingestion.metadata import MetadataEnricher

logger = get_logger(__name__)


class IngestionPipeline:

    @staticmethod
    def run(source: str, metadata: dict, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Executes the full ingestion pipeline: Load -> Enrich Metadata -> Chunk.
        """
        start = perf_counter()
        logger.info(
            "Ingestion pipeline started",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )

        try:
            # 1. Load documents
            raw_documents = DocumentLoader.load(source)

            # 2. Add metadata
            enriched_documents = MetadataEnricher.enrich(raw_documents, metadata)

            # 3. Chunk documents
            document_chunks = DocumentChunker.split(
                enriched_documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception:
            logger.exception(
                "Ingestion pipeline failed",
                extra={
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            raise

        logger.info(
            "Ingestion pipeline completed",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
                "chunk_count": len(document_chunks),
            },
        )
        return document_chunks

if __name__ == "__main__":
    # Example usage:
    chunks = IngestionPipeline.run(
        source="data/raw/company_handbook.txt",
        metadata={"category": "documentation", "author": "admin"},
        chunk_size=500,
        chunk_overlap=50,
    )
    logger.info(
        "Ingestion pipeline example completed",
        extra={
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "chunk_count": len(chunks),
        },
    )

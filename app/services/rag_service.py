from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from app.config.constants import CHUNK_OVERLAP, CHUNK_SIZE
from app.config.settings import settings
from app.chains.rag_chain import RAGChain
from app.core.exceptions import LLMError, RAGError, VectorStoreError
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.llm.groq import get_groq_llm
from app.reranking.bge_reranker import BGEReranker
from app.retrieval.retriever import Retriever

logger = get_logger(__name__)


@dataclass(frozen=True)
class SourceItem:
    source: str | None
    score: float | None


@dataclass(frozen=True)
class ChatResult:
    answer: str
    question: str
    rewritten_question: str
    sources: list[SourceItem]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IngestionResult:
    status: str
    documents_indexed: int
    chunks_created: int
    source: str


@dataclass(frozen=True)
class DocumentRecord:
    id: str | None
    source: str | None
    metadata: dict[str, Any]
    content: str


class RAGService:
    def __init__(self, retriever: Retriever | None = None, chain: RAGChain | None = None) -> None:
        self.retriever = retriever or Retriever(
            reranker=BGEReranker(),
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        self.chain = chain or RAGChain(retriever=self.retriever, llm=get_groq_llm(), top_k=settings.RAG_TOP_K)
        self.model_name = settings.LLM_MODEL

    def ask(self, question: str, rewrite: bool | None = None, top_k: int | None = None) -> ChatResult:
        should_rewrite = settings.RAG_REWRITE_QUERY if rewrite is None else rewrite
        limit = top_k or settings.RAG_TOP_K
        start = perf_counter()
        logger.info(
            "RAG request received",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )
        try:
            response = self.chain.ask(question=question, rewrite=should_rewrite, top_k=limit)
        except Exception as error:
            logger.exception(
                "RAG request failed",
                extra={
                    "error_code": "RAG_ERROR",
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            if error.__class__.__name__.endswith("NotFoundError") or "model" in str(error).lower():
                raise LLMError("The configured LLM model is unavailable.") from error
            raise RAGError("The RAG request failed.") from error

        latency_ms = int((perf_counter() - start) * 1000)
        input_tokens = None
        output_tokens = None
        if response.llm_metadata:
            input_tokens = response.llm_metadata.get("input_tokens")
            output_tokens = response.llm_metadata.get("output_tokens")

        sources = [
            SourceItem(
                source=document.metadata.get("source") or document.metadata.get("id"),
                score=self._extract_score(document),
            )
            for document in response.context_documents
        ]
        metadata: dict[str, Any] = {
            "model": self.model_name,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        logger.info(
            "RAG response completed",
            extra={
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "result_count": len(sources),
            },
        )

        return ChatResult(
            answer=response.answer,
            question=response.question,
            rewritten_question=response.rewritten_question,
            sources=sources,
            metadata=metadata,
        )

    def ingest_source(
        self,
        source: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        resolved_chunk_size = chunk_size or CHUNK_SIZE
        resolved_chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        start = perf_counter()
        try:
            chunks = IngestionPipeline.run(
                source=source,
                metadata=metadata or {},
                chunk_size=resolved_chunk_size,
                chunk_overlap=resolved_chunk_overlap,
            )
            self.chain.index_documents(chunks)
        except Exception as error:
            logger.exception(
                "Document ingestion failed",
                extra={
                    "error_code": "VECTOR_STORE_ERROR",
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            raise VectorStoreError("Document ingestion failed.") from error

        latency_ms = int((perf_counter() - start) * 1000)
        logger.info(
            "Document ingestion completed",
            extra={
                "latency_ms": latency_ms,
                "input_tokens": None,
                "output_tokens": None,
                "chunk_count": len(chunks),
            },
        )
        return IngestionResult(
            status="success",
            documents_indexed=1,
            chunks_created=len(chunks),
            source=source,
        )

    def list_documents(self) -> list[DocumentRecord]:
        try:
            documents = self.retriever.list_documents()
        except Exception as error:
            logger.exception("Document listing failed", extra={"error_code": "VECTOR_STORE_ERROR"})
            raise VectorStoreError("Document listing failed.") from error

        records: list[DocumentRecord] = []
        for document in documents:
            records.append(
                DocumentRecord(
                    id=document.metadata.get("id"),
                    source=document.metadata.get("source") or document.metadata.get("id"),
                    metadata=dict(document.metadata or {}),
                    content=document.page_content,
                )
            )
        return records

    def delete_document(self, document_id: str) -> int:
        try:
            return self.retriever.delete_documents([document_id])
        except Exception as error:
            logger.exception("Document deletion failed", extra={"error_code": "VECTOR_STORE_ERROR"})
            raise VectorStoreError("Document deletion failed.") from error

    @staticmethod
    def _extract_score(document: Document) -> float | None:
        score = document.metadata.get("rerank_score")
        if score is None:
            score = document.metadata.get("hybrid_score")
        if score is None:
            score = document.metadata.get("score")
        if score is None:
            score = document.metadata.get("distance")
        return score

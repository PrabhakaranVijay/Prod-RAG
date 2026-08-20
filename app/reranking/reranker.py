from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from langchain_core.documents import Document

from app.config.constants import RERANK_TOP_K
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RankedDocument:
    document: Document
    score: float


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int = RERANK_TOP_K,
    ) -> list[Document]:
        start = perf_counter()
        logger.info(
            "Reranking started",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )

        if not query.strip() or not documents:
            logger.info(
                "Reranking completed",
                extra={
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                    "document_count": 0,
                },
            )
            return []

        try:
            scores = self.score(query=query, documents=documents)
        except Exception:
            logger.exception(
                "Reranking failed",
                extra={
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            raise

        if len(scores) != len(documents):
            logger.error(
                "Reranking failed",
                extra={
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            raise ValueError("Reranker returned a score count that does not match the documents count.")

        ranked_documents = sorted(
            (
                RankedDocument(document=document, score=score)
                for document, score in zip(documents, scores, strict=True)
            ),
            key=lambda item: item.score,
            reverse=True,
        )

        results: list[Document] = []
        for rank, item in enumerate(ranked_documents[:top_k], start=1):
            metadata = dict(item.document.metadata or {})
            metadata["rerank_score"] = item.score
            metadata["rank"] = rank
            results.append(
                Document(
                    page_content=item.document.page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Reranking completed",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
                "document_count": len(results),
            },
        )
        return results

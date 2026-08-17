from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document

from app.config.constants import RERANK_TOP_K


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
        if not query.strip() or not documents:
            return []

        scores = self.score(query=query, documents=documents)
        if len(scores) != len(documents):
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

        return results

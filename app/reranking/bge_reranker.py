from __future__ import annotations

from time import perf_counter
from typing import Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config.settings import settings
from app.core.logging import get_logger
from app.reranking.reranker import Reranker

logger = get_logger(__name__)


class BGEReranker(Reranker):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device
        self.max_length = max_length

        start = perf_counter()
        logger.info(
            "Loading BGE reranker model",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        logger.info(
            "BGE reranker model loaded",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
            },
        )

    def score(self, query: str, documents: Sequence[Document]) -> list[float]:
        if not query.strip() or not documents:
            return []

        pairs = [[query, document.page_content] for document in documents]
        raw_scores = self.model.predict(pairs)

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        return [float(score) for score in raw_scores]

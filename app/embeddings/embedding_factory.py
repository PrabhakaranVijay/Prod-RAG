from __future__ import annotations

from time import perf_counter
from typing import Any

from app.core.logging import get_logger
from app.embeddings.bge_embeddings import BGEEmbeddings

logger = get_logger(__name__)


class LoggedEmbeddings:
    def __init__(self, embeddings: Any) -> None:
        self._embeddings = embeddings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)

    def embed_query(self, text: str) -> Any:
        start = perf_counter()
        logger.info(
            "Embedding generation started",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )
        try:
            embedding = self._embeddings.embed_query(text)
        except Exception:
            logger.exception(
                "Embedding generation failed",
                extra={
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            raise

        logger.info(
            "Embedding generation completed",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
            },
        )
        return embedding

    def embed_documents(self, texts: list[str]) -> Any:
        start = perf_counter()
        logger.info(
            "Embedding generation started",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )
        try:
            embeddings = self._embeddings.embed_documents(texts)
        except Exception:
            logger.exception(
                "Embedding generation failed",
                extra={
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            raise

        logger.info(
            "Embedding generation completed",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
            },
        )
        return embeddings


def ensure_logged_embeddings(embeddings: Any) -> LoggedEmbeddings:
    if isinstance(embeddings, LoggedEmbeddings):
        return embeddings
    return LoggedEmbeddings(embeddings)


class EmbeddingFactory:

    @staticmethod
    def create_embedding(
        provider: str = "bge"
    ):

        if provider.lower() == "bge":
            return ensure_logged_embeddings(BGEEmbeddings().get_embedding_model())

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

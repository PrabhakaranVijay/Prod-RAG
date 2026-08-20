from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

from langchain_core.documents import Document

from app.config.constants import TOP_K
from app.core.logging import get_logger
from app.vectorstore.chroma import ChromaVectorStore

logger = get_logger(__name__)


class VectorSearchRetriever:
	def __init__(
		self,
		vector_store: ChromaVectorStore | None = None,
		**vector_store_kwargs: Any,
	) -> None:
		self.vector_store = vector_store or ChromaVectorStore(**vector_store_kwargs)

	def add_documents(
		self,
		documents: Sequence[Document],
		ids: Sequence[str] | None = None,
	) -> list[str]:
		return self.vector_store.add_documents(documents, ids=ids)

	def add_source(
		self,
		source: str,
		metadata: dict[str, Any] | None = None,
		chunk_size: int | None = None,
		chunk_overlap: int | None = None,
	) -> list[str]:
		kwargs: dict[str, Any] = {"source": source, "metadata": metadata}
		if chunk_size is not None:
			kwargs["chunk_size"] = chunk_size
		if chunk_overlap is not None:
			kwargs["chunk_overlap"] = chunk_overlap
		return self.vector_store.build_index_from_source(**kwargs)

	def search(self, query: str, top_k: int = TOP_K) -> list[Document]:
		start = perf_counter()
		logger.info(
			"Vector retrieval started",
			extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
		)
		try:
			results = self.vector_store.search(query=query, top_k=top_k)
		except Exception:
			logger.exception(
				"Vector retrieval failed",
				extra={
					"latency_ms": int((perf_counter() - start) * 1000),
					"input_tokens": None,
					"output_tokens": None,
				},
			)
			raise

		logger.info(
			"Vector retrieval completed",
			extra={
				"latency_ms": int((perf_counter() - start) * 1000),
				"input_tokens": None,
				"output_tokens": None,
				"result_count": len(results),
			},
		)
		return results

	def list_documents(self) -> list[Document]:
		return self.vector_store.list_documents()

	def delete_documents(self, ids: Sequence[str]) -> int:
		return self.vector_store.delete_documents(ids)

	def count(self) -> int:
		return self.vector_store.count()

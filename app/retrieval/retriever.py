from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from app.config.constants import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.reranking.reranker import Reranker
from app.retrieval.vector_search import VectorSearchRetriever

logger = get_logger(__name__)


@dataclass
class RetrieverConfig:
	mode: str = "hybrid"
	top_k: int = TOP_K


class Retriever:
	def __init__(
		self,
		config: RetrieverConfig | None = None,
		vector_search: VectorSearchRetriever | None = None,
		bm25_retriever: BM25Retriever | None = None,
		hybrid_retriever: HybridRetriever | None = None,
		reranker: Reranker | None = None,
		**vector_store_kwargs: Any,
	) -> None:
		self.config = config or RetrieverConfig()
		self.vector_search = vector_search or VectorSearchRetriever(**vector_store_kwargs)
		self.bm25_retriever = bm25_retriever or BM25Retriever()
		self.hybrid_retriever = hybrid_retriever or HybridRetriever(
			vector_search=self.vector_search,
			bm25_retriever=self.bm25_retriever,
			reranker=reranker,
		)

	def index_documents(self, documents: list[Document]) -> list[str]:
		ids = self.vector_search.add_documents(documents)
		self.bm25_retriever.index_documents(documents)
		return ids

	def list_documents(self) -> list[Document]:
		return self.vector_search.list_documents()

	def delete_documents(self, ids: list[str]) -> int:
		deleted_count = self.vector_search.delete_documents(ids)
		return deleted_count

	def count_documents(self) -> int:
		return self.vector_search.count()

	def index_source(
		self,
		source: str,
		metadata: dict[str, Any] | None = None,
		chunk_size: int | None = None,
		chunk_overlap: int | None = None,
	) -> list[str]:
		resolved_chunk_size = chunk_size or CHUNK_SIZE
		resolved_chunk_overlap = chunk_overlap or CHUNK_OVERLAP
		chunks = IngestionPipeline.run(
			source=source,
			metadata=metadata or {},
			chunk_size=resolved_chunk_size,
			chunk_overlap=resolved_chunk_overlap,
		)
		ids = self.vector_search.add_documents(
			chunks,
		)
		self.bm25_retriever.index_documents(chunks)
		return ids

	def search(self, query: str, top_k: int | None = None) -> list[Document]:
		limit = top_k or self.config.top_k
		mode = self.config.mode.lower()

		start = perf_counter()
		logger.info(
			"Retrieval started",
			extra={
				"latency_ms": None,
				"input_tokens": None,
				"output_tokens": None,
				"retrieval_mode": mode,
			},
		)

		try:
			if mode == "vector":
				results = self.vector_search.search(query=query, top_k=limit)
			elif mode == "bm25":
				results = self.bm25_retriever.search(query=query, top_k=limit)
			elif mode == "hybrid":
				results = self.hybrid_retriever.search(query=query, top_k=limit)
			else:
				raise ValueError(f"Unsupported retrieval mode: {self.config.mode}")
		except Exception:
			logger.exception(
				"Retrieval failed",
				extra={
					"latency_ms": int((perf_counter() - start) * 1000),
					"input_tokens": None,
					"output_tokens": None,
					"retrieval_mode": mode,
				},
			)
			raise

		logger.info(
			"Retrieval completed",
			extra={
				"latency_ms": int((perf_counter() - start) * 1000),
				"input_tokens": None,
				"output_tokens": None,
				"retrieval_mode": mode,
				"result_count": len(results),
			},
		)
		return results

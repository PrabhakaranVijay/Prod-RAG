from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from app.config.constants import TOP_K
from app.retrieval.bm25 import BM25Retriever
from app.reranking.reranker import Reranker
from app.retrieval.vector_search import VectorSearchRetriever


def _document_key(document: Document) -> str:
	metadata = document.metadata or {}
	return str(metadata.get("id") or metadata.get("source") or document.page_content)


@dataclass(frozen=True)
class HybridScore:
	document: Document
	score: float
	vector_rank: int | None = None
	bm25_rank: int | None = None


class HybridRetriever:
	def __init__(
		self,
		vector_search: VectorSearchRetriever,
		bm25_retriever: BM25Retriever,
		reranker: Reranker | None = None,
		vector_weight: float = 0.5,
		bm25_weight: float = 0.5,
		rrf_k: float = 60.0,
	) -> None:
		self.vector_search = vector_search
		self.bm25_retriever = bm25_retriever
		self.reranker = reranker
		self.vector_weight = vector_weight
		self.bm25_weight = bm25_weight
		self.rrf_k = rrf_k

	def search(self, query: str, top_k: int = TOP_K) -> list[Document]:
		vector_results = self.vector_search.search(query=query, top_k=top_k)
		bm25_results = self.bm25_retriever.search(query=query, top_k=top_k)

		fused: dict[str, HybridScore] = {}

		for rank, document in enumerate(vector_results, start=1):
			key = _document_key(document)
			score = self.vector_weight / (self.rrf_k + rank)
			fused[key] = HybridScore(
				document=document,
				score=score,
				vector_rank=rank,
			)

		for rank, document in enumerate(bm25_results, start=1):
			key = _document_key(document)
			score = self.bm25_weight / (self.rrf_k + rank)
			existing = fused.get(key)
			if existing is None:
				fused[key] = HybridScore(
					document=document,
					score=score,
					bm25_rank=rank,
				)
			else:
				metadata = dict(existing.document.metadata or {})
				metadata.update(document.metadata or {})
				merged_document = Document(
					page_content=existing.document.page_content,
					metadata=metadata,
				)
				fused[key] = HybridScore(
					document=merged_document,
					score=existing.score + score,
					vector_rank=existing.vector_rank,
					bm25_rank=rank,
				)

		ranked = sorted(
			fused.values(),
			key=lambda item: (
				-item.score,
				item.vector_rank or 10**9,
				item.bm25_rank or 10**9,
			),
		)

		results: list[Document] = []
		for output_rank, item in enumerate(ranked[:top_k], start=1):
			metadata = dict(item.document.metadata or {})
			metadata["hybrid_score"] = item.score
			metadata["rank"] = output_rank
			metadata["vector_rank"] = item.vector_rank
			metadata["bm25_rank"] = item.bm25_rank
			results.append(Document(page_content=item.document.page_content, metadata=metadata))

		if self.reranker is None:
			return results

		return self.reranker.rerank(query=query, documents=results, top_k=top_k)

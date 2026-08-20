from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from app.config.constants import TOP_K
from app.config.prompts import QUERY_REWRITE_PROMPT, RAG_SYSTEM_PROMPT
from app.core.logging import get_logger
from app.llm.groq import get_groq_llm
from app.retrieval.retriever import Retriever

logger = get_logger(__name__)


@dataclass(frozen=True)
class RAGResponse:
	question: str
	rewritten_question: str
	answer: str
	context_documents: list[Document]
	llm_metadata: dict[str, Any] | None = None


class RAGChain:
	def __init__(
		self,
		retriever: Retriever | None = None,
		llm: Any | None = None,
		top_k: int = TOP_K,
	) -> None:
		self.retriever = retriever or Retriever()
		self.llm = llm or get_groq_llm()
		self.top_k = top_k

	@staticmethod
	def _as_text(response: Any) -> str:
		content = getattr(response, "content", response)
		return str(content).strip()

	@staticmethod
	def _token_value(usage: dict[str, Any], *keys: str) -> int | None:
		for key in keys:
			value = usage.get(key)
			if value is not None:
				try:
					return int(value)
				except (TypeError, ValueError):
					return None
		return None

	@classmethod
	def _extract_usage(cls, response: Any) -> dict[str, Any] | None:
		usage = getattr(response, "usage_metadata", None)
		if usage is None:
			response_metadata = getattr(response, "response_metadata", None)
			if isinstance(response_metadata, dict):
				usage = response_metadata.get("token_usage") or response_metadata.get("usage")

		if usage is None:
			return None

		if hasattr(usage, "model_dump"):
			usage = usage.model_dump()

		if isinstance(usage, dict):
			return {
				"input_tokens": cls._token_value(usage, "input_tokens", "prompt_tokens"),
				"output_tokens": cls._token_value(usage, "output_tokens", "completion_tokens"),
			}

		return None

	@staticmethod
	def _format_document(document: Document, index: int) -> str:
		metadata = document.metadata or {}
		source = metadata.get("source") or metadata.get("id") or f"doc-{index}"
		score = metadata.get("rerank_score") or metadata.get("hybrid_score") or metadata.get("score") or metadata.get("distance")
		header = f"[{index}] source={source}"
		if score is not None:
			header = f"{header} score={score}"
		return f"{header}\n{document.page_content.strip()}"

	def rewrite_query(self, question: str) -> str:
		cleaned_question = question.strip()
		if not cleaned_question:
			return cleaned_question

		start = perf_counter()
		logger.info("Query rewriting started", extra={"latency_ms": None, "input_tokens": None, "output_tokens": None})
		prompt = QUERY_REWRITE_PROMPT.format(query=cleaned_question).strip()
		try:
			response = self.llm.invoke(prompt)
		except Exception:
			logger.exception(
				"Query rewriting failed",
				extra={
					"latency_ms": int((perf_counter() - start) * 1000),
					"input_tokens": None,
					"output_tokens": None,
				},
			)
			raise

		rewritten_question = self._as_text(response) or cleaned_question
		usage = self._extract_usage(response)
		logger.info(
			"Query rewriting completed",
			extra={
				"latency_ms": int((perf_counter() - start) * 1000),
				"input_tokens": None if usage is None else usage.get("input_tokens"),
				"output_tokens": None if usage is None else usage.get("output_tokens"),
			},
		)
		return rewritten_question

	def retrieve(
		self,
		question: str,
		top_k: int | None = None,
		rewrite: bool = True,
		search_question: str | None = None,
	) -> list[Document]:
		search_question = search_question or (self.rewrite_query(question) if rewrite else question.strip())
		return self.retriever.search(query=search_question, top_k=top_k or self.top_k)

	def build_context(self, documents: list[Document]) -> str:
		if not documents:
			return ""

		return "\n\n".join(
			self._format_document(document, index)
			for index, document in enumerate(documents, start=1)
		)

	def build_prompt(self, question: str, context: str) -> str:
		return (
			f"{RAG_SYSTEM_PROMPT.strip()}\n\n"
			f"Context:\n{context or 'No relevant context found.'}\n\n"
			f"Question: {question.strip()}\n"
			"Answer:"
		).strip()

	def generate_answer(self, question: str, context: str) -> tuple[str, dict[str, Any] | None]:
		start = perf_counter()
		logger.info("LLM generation started", extra={"latency_ms": None, "input_tokens": None, "output_tokens": None})
		try:
			prompt = self.build_prompt(question=question, context=context)
			response = self.llm.invoke(prompt)
		except Exception:
			logger.exception(
				"LLM generation failed",
				extra={
					"latency_ms": int((perf_counter() - start) * 1000),
					"input_tokens": None,
					"output_tokens": None,
				},
			)
			raise

		answer = self._as_text(response)
		usage = self._extract_usage(response)
		logger.info(
			"LLM generation completed",
			extra={
				"latency_ms": int((perf_counter() - start) * 1000),
				"input_tokens": None if usage is None else usage.get("input_tokens"),
				"output_tokens": None if usage is None else usage.get("output_tokens"),
			},
		)
		return answer, usage

	def answer(self, question: str, top_k: int | None = None, rewrite: bool = True) -> str:
		documents = self.retrieve(question=question, top_k=top_k, rewrite=rewrite)
		context = self.build_context(documents)
		answer, _ = self.generate_answer(question=question, context=context)
		return answer

	def ask(self, question: str, top_k: int | None = None, rewrite: bool = True) -> RAGResponse:
		rewritten_question = self.rewrite_query(question) if rewrite else question.strip()
		documents = self.retrieve(
			question=question,
			top_k=top_k,
			rewrite=False,
			search_question=rewritten_question,
		)
		context = self.build_context(documents)
		answer, usage = self.generate_answer(question=question, context=context)
		return RAGResponse(
			question=question.strip(),
			rewritten_question=rewritten_question,
			answer=answer,
			context_documents=documents,
			llm_metadata=usage,
		)

	def index_source(
		self,
		source: str,
		metadata: dict[str, Any] | None = None,
		chunk_size: int | None = None,
		chunk_overlap: int | None = None,
	) -> list[str]:
		return self.retriever.index_source(
			source=source,
			metadata=metadata,
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
		)

	def index_documents(self, documents: list[Document]) -> list[str]:
		return self.retriever.index_documents(documents)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.config.constants import TOP_K
from app.config.prompts import QUERY_REWRITE_PROMPT, RAG_SYSTEM_PROMPT
from app.llm.groq import get_groq_llm
from app.retrieval.retriever import Retriever


@dataclass(frozen=True)
class RAGResponse:
	question: str
	rewritten_question: str
	answer: str
	context_documents: list[Document]


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
	def _format_document(document: Document, index: int) -> str:
		metadata = document.metadata or {}
		source = metadata.get("source") or metadata.get("id") or f"doc-{index}"
		score = metadata.get("hybrid_score") or metadata.get("score") or metadata.get("distance")
		header = f"[{index}] source={source}"
		if score is not None:
			header = f"{header} score={score}"
		return f"{header}\n{document.page_content.strip()}"

	def rewrite_query(self, question: str) -> str:
		cleaned_question = question.strip()
		if not cleaned_question:
			return cleaned_question

		prompt = QUERY_REWRITE_PROMPT.format(query=cleaned_question).strip()
		response = self.llm.invoke(prompt)
		rewritten_question = self._as_text(response)
		return rewritten_question or cleaned_question

	def retrieve(self, question: str, top_k: int | None = None, rewrite: bool = True) -> list[Document]:
		search_question = self.rewrite_query(question) if rewrite else question
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

	def answer(self, question: str, top_k: int | None = None, rewrite: bool = True) -> str:
		documents = self.retrieve(question=question, top_k=top_k, rewrite=rewrite)
		context = self.build_context(documents)
		prompt = self.build_prompt(question=question, context=context)
		response = self.llm.invoke(prompt)
		return self._as_text(response)

	def ask(self, question: str, top_k: int | None = None, rewrite: bool = True) -> RAGResponse:
		rewritten_question = self.rewrite_query(question) if rewrite else question
		documents = self.retriever.search(query=rewritten_question, top_k=top_k or self.top_k)
		context = self.build_context(documents)
		prompt = self.build_prompt(question=question, context=context)
		answer = self._as_text(self.llm.invoke(prompt))
		return RAGResponse(
			question=question,
			rewritten_question=rewritten_question,
			answer=answer,
			context_documents=documents,
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

	def index_documents(self, documents: list[Document]) -> None:
		self.retriever.index_documents(documents)


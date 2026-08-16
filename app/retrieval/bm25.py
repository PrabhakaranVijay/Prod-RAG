from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from langchain_core.documents import Document

from app.config.constants import TOP_K

TokenizeFn = Callable[[str], list[str]]

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def default_tokenizer(text: str) -> list[str]:
	return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class RankedDocument:
	document: Document
	score: float
	rank: int


class BM25Retriever:
	def __init__(
		self,
		documents: Sequence[Document] | None = None,
		tokenizer: TokenizeFn = default_tokenizer,
		k1: float = 1.5,
		b: float = 0.75,
	) -> None:
		self.tokenizer = tokenizer
		self.k1 = k1
		self.b = b
		self.documents: list[Document] = []
		self.term_frequencies: list[Counter[str]] = []
		self.document_lengths: list[int] = []
		self.document_frequency: Counter[str] = Counter()
		self.average_document_length = 0.0

		if documents:
			self.index_documents(documents)

	def index_documents(self, documents: Sequence[Document]) -> None:
		self.documents = list(documents)
		self.term_frequencies = []
		self.document_lengths = []
		self.document_frequency = Counter()

		for document in self.documents:
			tokens = self.tokenizer(document.page_content)
			frequencies = Counter(tokens)
			self.term_frequencies.append(frequencies)
			self.document_lengths.append(len(tokens))
			self.document_frequency.update(frequencies.keys())

		total_length = sum(self.document_lengths)
		self.average_document_length = (
			total_length / len(self.document_lengths)
			if self.document_lengths
			else 0.0
		)

	def _score(self, query_tokens: list[str], document_index: int) -> float:
		if not self.documents:
			return 0.0

		document_length = self.document_lengths[document_index] or 1
		frequencies = self.term_frequencies[document_index]
		score = 0.0
		total_documents = len(self.documents)

		for token in query_tokens:
			document_frequency = self.document_frequency.get(token, 0)
			if document_frequency == 0:
				continue

			inverse_document_frequency = math.log(
				1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
			)
			term_frequency = frequencies.get(token, 0)
			denominator = term_frequency + self.k1 * (
				1 - self.b + self.b * document_length / max(self.average_document_length, 1.0)
			)
			score += inverse_document_frequency * (
				term_frequency * (self.k1 + 1)
			) / max(denominator, 1e-9)

		return score

	def search(self, query: str, top_k: int = TOP_K) -> list[Document]:
		if not query.strip() or not self.documents:
			return []

		query_tokens = self.tokenizer(query)
		ranked_documents = sorted(
			(
				RankedDocument(
					document=document,
					score=self._score(query_tokens, index),
					rank=index + 1,
				)
				for index, document in enumerate(self.documents)
			),
			key=lambda item: (item.score, -item.rank),
			reverse=True,
		)

		results: list[Document] = []
		for output_rank, item in enumerate(ranked_documents[:top_k], start=1):
			metadata = dict(item.document.metadata or {})
			metadata["score"] = item.score
			metadata["rank"] = output_rank
			metadata["source_rank"] = item.rank
			results.append(Document(page_content=item.document.page_content, metadata=metadata))

		return results


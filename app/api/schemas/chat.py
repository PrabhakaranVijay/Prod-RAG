from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config.constants import TOP_K


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question to send through the RAG pipeline.")
    rewrite: bool = Field(default=True, description="Whether to rewrite the query before retrieval.")
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="Maximum number of documents to retrieve.")

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: Any) -> str:
        question = str(value or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        return question


class ChatSource(BaseModel):
    source: str | None = None
    score: float | None = None


class ChatMetadata(BaseModel):
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatResponse(BaseModel):
    answer: str
    question: str
    rewritten_question: str
    sources: list[ChatSource]
    metadata: ChatMetadata

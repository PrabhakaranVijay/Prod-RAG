from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    source: str = Field(..., description="Local file path or supported URL to ingest.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int | None = Field(default=None, ge=1, le=5000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)

    @field_validator("source", mode="before")
    @classmethod
    def strip_source(cls, value: Any) -> str:
        source = str(value or "").strip()
        if not source:
            raise ValueError("source must not be empty")
        return source


class DocumentItem(BaseModel):
    id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]


class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    source: str


class DeleteDocumentResponse(BaseModel):
    status: str
    document_id: str
    deleted_count: int

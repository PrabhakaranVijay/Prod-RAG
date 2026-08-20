from __future__ import annotations

from fastapi import Request

from app.core.exceptions import ServiceUnavailableError
from app.services.rag_service import RAGService


def get_rag_service(request: Request) -> RAGService:
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise ServiceUnavailableError("The RAG service is not ready.")

    return service

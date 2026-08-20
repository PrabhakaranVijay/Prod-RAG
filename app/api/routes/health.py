from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check", description="Returns a basic liveness status.")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check", description="Indicates whether the RAG components initialized successfully.")
async def ready(request: Request, response: Response) -> dict[str, str]:
    initialization_error = getattr(request.app.state, "initialization_error", None)
    service = getattr(request.app.state, "rag_service", None)

    if initialization_error is not None or service is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready"}

    return {"status": "ready"}

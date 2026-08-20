from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, Request

from app.api.schemas.chat import ChatMetadata, ChatRequest, ChatResponse, ChatSource
from app.dependencies import get_rag_service
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Ask a question", description="Run the RAG pipeline against the indexed knowledge base.")
async def chat(request: ChatRequest, rag_service: RAGService = Depends(get_rag_service)) -> ChatResponse:
    start = perf_counter()
    result = rag_service.ask(
        question=request.question,
        rewrite=request.rewrite,
        top_k=request.top_k,
    )
    latency_ms = int((perf_counter() - start) * 1000)

    return ChatResponse(
        answer=result.answer,
        question=result.question,
        rewritten_question=result.rewritten_question,
        sources=[ChatSource(source=item.source, score=item.score) for item in result.sources],
        metadata=ChatMetadata(
            model=result.metadata.get("model"),
            latency_ms=latency_ms,
            input_tokens=result.metadata.get("input_tokens"),
            output_tokens=result.metadata.get("output_tokens"),
        ),
    )

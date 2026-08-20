from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.documents import (
    DeleteDocumentResponse,
    DocumentItem,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
)
from app.dependencies import get_rag_service
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document", description="Load, chunk, and index a local file or supported URL into the vector store.")
async def ingest_documents(payload: IngestRequest, rag_service: RAGService = Depends(get_rag_service)) -> IngestResponse:
    result = rag_service.ingest_source(
        source=payload.source,
        metadata=payload.metadata,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    return IngestResponse(
        status=result.status,
        documents_indexed=result.documents_indexed,
        chunks_created=result.chunks_created,
        source=result.source,
    )


@router.get("", response_model=DocumentListResponse, summary="List indexed documents", description="Return the current indexed documents from the vector store.")
async def list_documents(rag_service: RAGService = Depends(get_rag_service)) -> DocumentListResponse:
    documents = rag_service.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentItem(
                id=document.id,
                source=document.source,
                metadata=document.metadata,
                content=document.content,
            )
            for document in documents
        ]
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse, summary="Delete a document", description="Remove a document from the vector store if the backend supports deletion by id.")
async def delete_document(document_id: str, rag_service: RAGService = Depends(get_rag_service)) -> DeleteDocumentResponse:
    deleted_count = rag_service.delete_document(document_id)
    return DeleteDocumentResponse(status="deleted", document_id=document_id, deleted_count=deleted_count)

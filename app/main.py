from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from time import perf_counter

import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.exceptions import AppError, InitializationError
from app.core.logging import configure_logging, get_logger
from app.reranking.bge_reranker import BGEReranker
from app.retrieval.retriever import Retriever
from app.services.rag_service import RAGService

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = None
    app.state.initialization_error = None

    try:
        retriever = Retriever(
            reranker=BGEReranker(),
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        app.state.rag_service = RAGService(retriever=retriever)
        logger.info(
            "Application initialized",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )
    except Exception as error:  # pragma: no cover - startup failure is validated via /ready
        app.state.initialization_error = error
        logger.exception(
            "Application initialization failed",
            extra={"error_code": InitializationError.code},
        )

    yield


app = FastAPI(
    title="Prod-RAG API",
    version="0.1.0",
    description="Production-ready FastAPI server for the Prod-RAG application.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = perf_counter()

    logger.info(
        "API request received",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "API request failed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
            },
        )
        raise

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "API request completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": int((perf_counter() - start) * 1000),
            "input_tokens": None,
            "output_tokens": None,
        },
    )
    return response


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
            }
        },
    )


@app.get("/", summary="Root endpoint", description="Returns basic API metadata.")
async def root() -> dict[str, str]:
    return {"name": "Prod-RAG API", "status": "ok", "version": "0.1.0"}


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)



def main() -> None:
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations


class AppError(Exception):
    code = "APP_ERROR"
    status_code = 500

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ValidationAppError(AppError):
    code = "VALIDATION_ERROR"
    status_code = 422


class InitializationError(AppError):
    code = "INITIALIZATION_ERROR"
    status_code = 503


class ServiceUnavailableError(AppError):
    code = "SERVICE_UNAVAILABLE"
    status_code = 503


class RAGError(AppError):
    code = "RAG_ERROR"
    status_code = 500


class VectorStoreError(AppError):
    code = "VECTOR_STORE_ERROR"
    status_code = 500


class LLMError(AppError):
    code = "LLM_ERROR"
    status_code = 502


class DocumentNotFoundError(AppError):
    code = "DOCUMENT_NOT_FOUND"
    status_code = 404


class UnsupportedFileTypeError(AppError):
    code = "UNSUPPORTED_FILE_TYPE"
    status_code = 400


class DocumentParseError(AppError):
    code = "DOCUMENT_PARSE_ERROR"
    status_code = 400


class EmbeddingError(AppError):
    code = "EMBEDDING_ERROR"
    status_code = 500


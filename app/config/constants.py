from app.core.config import settings

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP

TOP_K = settings.RAG_TOP_K
RERANK_TOP_K = 5

DEFAULT_COLLECTION = "documents"

MAX_CONTEXT_TOKENS = 8000
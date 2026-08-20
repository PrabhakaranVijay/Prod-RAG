from app.core.config import settings


MODELS = {
    "chat": settings.GROQ_MODEL,
    "embedding": settings.EMBEDDING_MODEL,
    "reranker": settings.RERANKER_MODEL,
}
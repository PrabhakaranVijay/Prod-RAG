from app.config.settings import settings


MODELS = {
    "chat": settings.LLM_MODEL,
    "embedding": settings.EMBEDDING_MODEL,
    "reranker": settings.RERANKER_MODEL,
}
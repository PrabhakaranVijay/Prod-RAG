from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ELIZA"
    USER_AGENT: str = "ELIZA/1.0"

    GROQ_API_KEY: str | None = None
    HUGGINGFACE_API_KEY: str | None = None

    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "rag_db"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
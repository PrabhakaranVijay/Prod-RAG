from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Prod-RAG"
    USER_AGENT: str = "Prod-RAG/1.0"

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_MODEL", "LLM_MODEL"),
    )
    GROQ_TEMPERATURE: float = 0.0

    HF_TOKEN: str | None = None

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    CHROMA_PERSIST_DIRECTORY: str | None = None

    RAG_TOP_K: int = 5
    RAG_REWRITE_QUERY: bool = True

    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "rag_db"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    @property
    def LLM_MODEL(self) -> str | None:
        return self.GROQ_MODEL

    @LLM_MODEL.setter
    def LLM_MODEL(self, value: str | None) -> None:
        self.GROQ_MODEL = value


settings = Settings()

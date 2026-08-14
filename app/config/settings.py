from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ELIZA"
    USER_AGENT: str = "ELIZA/1.0"

    GROQ_API_KEY: str
    HUGGINGFACE_API_KEY: str | None = None

    LLM_MODEL: str = "llama-3.3-70b-versatile"

    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "rag_db"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"

settings = Settings()
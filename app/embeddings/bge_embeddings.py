from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import settings
from app.config.logging import logger


class BGEEmbeddings:

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        model_name = model_name or settings.EMBEDDING_MODEL

        logger.info(
            f"Loading embedding model: {model_name}"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": device
            },
            encode_kwargs={
                "normalize_embeddings": normalize_embeddings
            },
        )

    def get_embedding_model(self):
        return self.embeddings
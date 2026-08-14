from langchain_huggingface import HuggingFaceEmbeddings

from app.config.logging import logger


class BGEEmbeddings:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3", # 1,024 dimensions
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):

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
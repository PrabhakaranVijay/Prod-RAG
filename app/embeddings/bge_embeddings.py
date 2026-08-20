from time import perf_counter

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BGEEmbeddings:

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        model_name = model_name or settings.EMBEDDING_MODEL

        start = perf_counter()
        logger.info(
            "Loading embedding model",
            extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
        )

        model_kwargs = {"device": device}
        if settings.HF_TOKEN:
            model_kwargs["token"] = settings.HF_TOKEN

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
        logger.info(
            "Embedding model loaded",
            extra={
                "latency_ms": int((perf_counter() - start) * 1000),
                "input_tokens": None,
                "output_tokens": None,
            },
        )

    def get_embedding_model(self):
        return self.embeddings

from langchain_groq import ChatGroq

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_groq_llm():
    """
    Returns a ChatGroq instance with the specified model and temperature.
    """
    model = settings.LLM_MODEL
    if not model:
        raise ValueError("GROQ_MODEL must be configured before initializing ChatGroq.")

    logger.info(
        "Initializing ChatGroq",
        extra={"latency_ms": None, "input_tokens": None, "output_tokens": None},
    )
    return ChatGroq(
        model=model,
        temperature=settings.GROQ_TEMPERATURE,
        groq_api_key=settings.GROQ_API_KEY,
    )

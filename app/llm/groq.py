import logging

from langchain_groq import ChatGroq

from app.config.settings import settings


def get_groq_llm():
    """
    Returns a ChatGroq instance with the specified model and temperature.
    """
    logging.info(
        "Initializing ChatGroq with model '%s' and temperature 0.",
        settings.LLM_MODEL,
    )
    return ChatGroq(
        model=settings.LLM_MODEL,
        temperature=0,
    )
import logging

from langchain_groq import ChatGroq


def get_groq_llm():
    """
    Returns a ChatGroq instance with the specified model and temperature.
    """
    logging.info("Initializing ChatGroq with model 'llama-3.3-70b-versatile' and temperature 0.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
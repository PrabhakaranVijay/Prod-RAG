import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    UnstructuredFileLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

def load_text_file(file_path: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(file_path.encode())
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(f"Loaded document: {doc.page_content[:100]}...")  # Print first 100 characters
            print(f"Metadata: {doc.metadata}")

    finally:
        os.remove(temp_file_path)  # Clean up the temporary file

def load_web_page(url: str):
    loader = WebBaseLoader(url)
    documents = loader.load()

    for doc in documents:
        print(f"Loaded document: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"Metadata: {doc.metadata}")

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        print(f"Loaded document: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text file content for testing."
    load_text_file(sample_text)
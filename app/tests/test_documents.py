from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.api.schemas.documents import IngestRequest
from app.embeddings.embedding_factory import EmbeddingFactory
from app.vectorstore.chroma import ChromaVectorStore
from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c



def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "name" in json_data


def test_health_endpoints(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response_ready = client.get("/ready")
    assert response_ready.status_code in (200, 503)


def test_ingest_request_schema_validation():
    # Valid chunk configuration
    req = IngestRequest(source="test_data/sample.txt", chunk_size=500, chunk_overlap=100)
    assert req.chunk_size == 500
    assert req.chunk_overlap == 100

    # Invalid chunk configuration: overlap >= size should raise ValueError
    with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
        IngestRequest(source="test_data/sample.txt", chunk_size=1, chunk_overlap=1000)

    with pytest.raises(ValueError, match="source must not be empty"):
        IngestRequest(source="   ")


def test_ingest_document_api_validation_failure(client):
    # Test invalid overlap > size parameter via HTTP POST
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "source": "test_data/sample.txt",
            "metadata": {"test": True},
            "chunk_size": 1,
            "chunk_overlap": 1000,
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_ingest_document_missing_source(client):
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "metadata": {"test": True},
            "chunk_size": 500,
            "chunk_overlap": 100,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_ingest_document_non_existent_file(client):
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "source": "test_data/does_not_exist.txt",
            "metadata": {"test": True},
            "chunk_size": 500,
            "chunk_overlap": 100,
        },
    )
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_ingest_document_unsupported_file_type(client):
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "source": "test_data/sample.unsupported_ext",
            "metadata": {"test": True},
            "chunk_size": 500,
            "chunk_overlap": 100,
        },
    )
    assert response.status_code in (400, 404)
    data = response.json()
    assert data["error"]["code"] in ("UNSUPPORTED_FILE_TYPE", "DOCUMENT_NOT_FOUND")


def test_ingest_document_success_and_retrieval(client, tmp_path):
    # Test ingesting valid document
    response = client.post(
        "/api/v1/documents/ingest",
        json={
            "source": "test_data/sample.txt",
            "metadata": {"category": "test"},
            "chunk_size": 300,
            "chunk_overlap": 50,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_indexed"] == 1
    assert data["chunks_created"] > 0
    assert data["source"] == "test_data/sample.txt"

    # Verify listing documents
    list_response = client.get("/api/v1/documents")
    assert list_response.status_code == 200
    docs = list_response.json()["documents"]
    assert len(docs) > 0


def test_embedding_factory_independent():
    embedding_model = EmbeddingFactory.create_embedding("bge")
    vector = embedding_model.embed_query("Hello world")
    assert isinstance(vector, list)
    assert len(vector) > 0


def test_vector_store_independent(tmp_path):
    chroma_dir = str(tmp_path / "chroma_test")
    store = ChromaVectorStore(collection_name="test_collection", persist_directory=chroma_dir)
    store.build_index_from_source(source="test_data/sample.txt")
    results = store.search(query="Prod-RAG FastAPI", top_k=2)
    assert len(results) > 0
    assert "Prod-RAG" in results[0].page_content


def test_chat_endpoint_mocked(client):
    with patch("app.services.rag_service.get_groq_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Prod-RAG is a high-performance RAG system built with FastAPI."
        mock_response.usage_metadata = {"input_tokens": 50, "output_tokens": 20}
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/v1/chat",
            json={
                "question": "What is Prod-RAG?",
                "top_k": 3,
                "rewrite": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["question"] == "What is Prod-RAG?"


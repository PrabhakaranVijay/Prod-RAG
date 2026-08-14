from app.embeddings.embedding_factory import EmbeddingFactory


def test_embedding_vector_shape():
    embeddings = EmbeddingFactory.create_embedding()
    vector = embeddings.embed_query("What is RAG?")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)

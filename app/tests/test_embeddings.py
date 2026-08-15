from app.embeddings.embedding_factory import EmbeddingFactory


class _FakeEmbeddingModel:
    def embed_query(self, text: str):
        return [float(len(text)), 1.0]


class _FakeBGEEmbeddings:
    def get_embedding_model(self):
        return _FakeEmbeddingModel()


def test_embedding_vector_shape(monkeypatch):
    monkeypatch.setattr(
        "app.embeddings.embedding_factory.BGEEmbeddings",
        _FakeBGEEmbeddings,
    )

    embeddings = EmbeddingFactory.create_embedding()
    vector = embeddings.embed_query("What is RAG?")

    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)

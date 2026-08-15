from langchain_core.documents import Document

from app.vectorstore.chroma import ChromaVectorStore


class DummyEmbeddings:
    def embed_documents(self, texts):
        return [self._encode(text) for text in texts]

    def embed_query(self, text):
        return self._encode(text)

    @staticmethod
    def _encode(text):
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0]
        return [0.0, 0.0]


def test_chroma_vector_store_indexes_and_searches(tmp_path):
    store = ChromaVectorStore(
        collection_name="test_documents",
        embeddings=DummyEmbeddings(),
        persist_directory=str(tmp_path),
    )

    ids = store.add_documents(
        [
            Document(page_content="alpha handbook entry", metadata={"source": "alpha"}),
            Document(page_content="beta policy entry", metadata={"source": "beta"}),
        ],
        ids=["alpha-doc", "beta-doc"],
    )

    assert ids == ["alpha-doc", "beta-doc"]
    assert store.count() == 2

    results = store.search("alpha", top_k=1)

    assert len(results) == 1
    assert results[0].page_content == "alpha handbook entry"
    assert results[0].metadata["source"] == "alpha"
    assert results[0].metadata["id"] == "alpha-doc"
    assert results[0].metadata["rank"] == 1
    assert results[0].metadata["distance"] is not None


def test_build_index_from_source_uses_ingestion_pipeline(monkeypatch, tmp_path):
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            Document(
                page_content="alpha handbook chunk",
                metadata={"source": "handbook.txt", "category": kwargs["metadata"]["category"]},
            ),
            Document(
                page_content="beta handbook chunk",
                metadata={"source": "handbook.txt", "category": kwargs["metadata"]["category"]},
            ),
        ]

    monkeypatch.setattr("app.vectorstore.chroma.IngestionPipeline.run", fake_run)

    store = ChromaVectorStore(
        collection_name="source_documents",
        embeddings=DummyEmbeddings(),
        persist_directory=str(tmp_path),
    )

    ids = store.build_index_from_source(
        source="data/raw/company_handbook.txt",
        metadata={"category": "documentation"},
        chunk_size=250,
        chunk_overlap=25,
    )

    assert len(ids) == 2
    assert store.count() == 2
    assert captured_kwargs["source"] == "data/raw/company_handbook.txt"
    assert captured_kwargs["metadata"] == {"category": "documentation"}
    assert captured_kwargs["chunk_size"] == 250
    assert captured_kwargs["chunk_overlap"] == 25

    results = store.search("beta", top_k=1)

    assert results[0].page_content == "beta handbook chunk"
    assert results[0].metadata["category"] == "documentation"
from langchain_core.documents import Document

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.retriever import Retriever, RetrieverConfig
from app.retrieval.vector_search import VectorSearchRetriever


class FakeVectorStore:
    def __init__(self):
        self.documents = []
        self.source_calls = []
        self.search_results = []

    def add_documents(self, documents, ids=None):
        self.documents.extend(documents)
        if ids is not None:
            return list(ids)

        return [
            str(document.metadata.get("id") or f"doc-{index}")
            for index, document in enumerate(documents)
        ]

    def build_index_from_source(self, **kwargs):
        self.source_calls.append(kwargs)
        return ["source-1"]

    def search(self, query, top_k):
        return self.search_results[:top_k]


def test_vector_search_retriever_delegates_to_vector_store():
    fake_store = FakeVectorStore()
    retriever = VectorSearchRetriever(vector_store=fake_store)

    documents = [Document(page_content="alpha", metadata={"id": "a"})]

    ids = retriever.add_documents(documents, ids=["a"])
    source_ids = retriever.add_source(
        source="data/raw/company_handbook.txt",
        metadata={"category": "docs"},
        chunk_size=300,
        chunk_overlap=50,
    )
    results = retriever.search("alpha", top_k=1)

    assert ids == ["a"]
    assert fake_store.documents == documents
    assert source_ids == ["source-1"]
    assert fake_store.source_calls[0]["source"] == "data/raw/company_handbook.txt"
    assert fake_store.source_calls[0]["metadata"] == {"category": "docs"}
    assert fake_store.source_calls[0]["chunk_size"] == 300
    assert fake_store.source_calls[0]["chunk_overlap"] == 50
    assert results == []


def test_bm25_retriever_ranks_relevant_document_first():
    retriever = BM25Retriever(
        [
            Document(page_content="the cat sat on the mat", metadata={"id": "cat"}),
            Document(page_content="brown fox jumps fast", metadata={"id": "fox"}),
        ]
    )

    results = retriever.search("brown fox", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["id"] == "fox"
    assert results[0].metadata["score"] > 0
    assert results[0].metadata["rank"] == 1


def test_hybrid_retriever_combines_vector_and_bm25_rankings():
    fake_vector_store = FakeVectorStore()
    fake_vector_store.search_results = [
        Document(page_content="alpha handbook entry", metadata={"id": "alpha", "source": "vector"}),
        Document(page_content="beta handbook entry", metadata={"id": "beta", "source": "vector"}),
    ]
    vector_search = VectorSearchRetriever(vector_store=fake_vector_store)
    bm25_retriever = BM25Retriever(
        [
            Document(page_content="beta handbook entry", metadata={"id": "beta", "source": "bm25"}),
            Document(page_content="alpha handbook entry", metadata={"id": "alpha", "source": "bm25"}),
        ]
    )

    hybrid_retriever = HybridRetriever(
        vector_search=vector_search,
        bm25_retriever=bm25_retriever,
    )

    results = hybrid_retriever.search("handbook", top_k=2)

    assert [result.metadata["id"] for result in results] == ["alpha", "beta"]
    assert results[0].metadata["hybrid_score"] >= results[1].metadata["hybrid_score"]
    assert results[0].metadata["vector_rank"] == 1


def test_retriever_facade_dispatches_by_mode(monkeypatch):
    fake_vector_store = FakeVectorStore()
    fake_vector_store.search_results = [Document(page_content="vector result", metadata={"id": "vector"})]
    vector_search = VectorSearchRetriever(vector_store=fake_vector_store)
    bm25_retriever = BM25Retriever([Document(page_content="bm25 result", metadata={"id": "bm25"})])

    retriever = Retriever(
        config=RetrieverConfig(mode="hybrid", top_k=1),
        vector_search=vector_search,
        bm25_retriever=bm25_retriever,
    )

    hybrid_results = retriever.search("result")
    assert hybrid_results[0].metadata["id"] == "vector"

    retriever.config.mode = "vector"
    vector_results = retriever.search("result")
    assert vector_results[0].metadata["id"] == "vector"

    retriever.config.mode = "bm25"
    bm25_results = retriever.search("result")
    assert bm25_results[0].metadata["id"] == "bm25"


def test_retriever_index_source_populates_both_indexes(monkeypatch):
    fake_vector_store = FakeVectorStore()
    vector_search = VectorSearchRetriever(vector_store=fake_vector_store)
    bm25_retriever = BM25Retriever()

    monkeypatch.setattr(
        "app.retrieval.retriever.IngestionPipeline.run",
        lambda **kwargs: [
            Document(page_content="alpha", metadata={"id": "alpha"}),
            Document(page_content="beta", metadata={"id": "beta"}),
        ],
    )

    retriever = Retriever(
        vector_search=vector_search,
        bm25_retriever=bm25_retriever,
    )

    ids = retriever.index_source(
        source="data/raw/company_handbook.txt",
        metadata={"category": "docs"},
        chunk_size=250,
        chunk_overlap=25,
    )

    assert ids == ["alpha", "beta"]
    assert len(fake_vector_store.documents) == 2
    assert len(bm25_retriever.documents) == 2

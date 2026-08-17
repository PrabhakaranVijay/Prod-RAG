from langchain_core.documents import Document

from app.reranking.bge_reranker import BGEReranker


def test_bge_reranker_scores_and_orders_documents(monkeypatch):
    from app.reranking import bge_reranker

    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, device=None, max_length=None):
            captured["model_name"] = model_name
            captured["device"] = device
            captured["max_length"] = max_length

        def predict(self, pairs):
            captured["pairs"] = pairs
            return [0.2, 0.9]

    monkeypatch.setattr(bge_reranker, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(bge_reranker.settings, "RERANKER_MODEL", "env-reranker-model")

    reranker = BGEReranker(device="cpu", max_length=256)
    results = reranker.rerank(
        "what is the handbook",
        [
            Document(page_content="first document", metadata={"id": "doc-1"}),
            Document(page_content="second document", metadata={"id": "doc-2"}),
        ],
        top_k=1,
    )

    assert captured["model_name"] == "env-reranker-model"
    assert captured["device"] == "cpu"
    assert captured["max_length"] == 256
    assert captured["pairs"] == [
        ["what is the handbook", "first document"],
        ["what is the handbook", "second document"],
    ]
    assert len(results) == 1
    assert results[0].metadata["id"] == "doc-2"
    assert results[0].metadata["rerank_score"] == 0.9
    assert results[0].metadata["rank"] == 1


def test_reranker_rejects_mismatched_score_counts():
    class BrokenReranker(BGEReranker):
        def __init__(self):
            pass

        def score(self, query, documents):
            return [0.5]

    reranker = BrokenReranker()

    try:
        reranker.rerank(
            "query",
            [
                Document(page_content="doc-1"),
                Document(page_content="doc-2"),
            ],
            top_k=1,
        )
    except ValueError as error:
        assert "score count" in str(error)
    else:
        raise AssertionError("Expected rerank to raise ValueError")

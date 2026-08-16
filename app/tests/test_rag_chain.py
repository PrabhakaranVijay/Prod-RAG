from langchain_core.documents import Document

from app.chains.rag_chain import RAGChain, RAGResponse


class FakeRetriever:
    def __init__(self):
        self.index_source_calls = []
        self.index_documents_calls = []
        self.search_calls = []
        self.documents = [
            Document(page_content="alpha handbook entry", metadata={"id": "alpha", "source": "handbook"}),
            Document(page_content="beta policy entry", metadata={"id": "beta", "source": "policy"}),
        ]

    def index_source(self, **kwargs):
        self.index_source_calls.append(kwargs)
        return ["chunk-1", "chunk-2"]

    def index_documents(self, documents):
        self.index_documents_calls.append(documents)

    def search(self, query, top_k):
        self.search_calls.append((query, top_k))
        return self.documents[:top_k]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("Rewrite the query for better retrieval."):
            return type("Response", (), {"content": "rewritten query"})()
        return type("Response", (), {"content": "final answer"})()


def test_rag_chain_rewrites_retrieves_and_answers():
    retriever = FakeRetriever()
    llm = FakeLLM()
    chain = RAGChain(retriever=retriever, llm=llm, top_k=1)

    response = chain.ask("What is the handbook?", top_k=1)

    assert isinstance(response, RAGResponse)
    assert response.question == "What is the handbook?"
    assert response.rewritten_question == "rewritten query"
    assert response.answer == "final answer"
    assert len(response.context_documents) == 1
    assert retriever.search_calls == [("rewritten query", 1)]
    assert any("Context:" in prompt for prompt in llm.prompts)


def test_rag_chain_indexes_source_and_documents():
    retriever = FakeRetriever()
    llm = FakeLLM()
    chain = RAGChain(retriever=retriever, llm=llm)

    source_ids = chain.index_source(
        source="data/raw/company_handbook.txt",
        metadata={"category": "docs"},
        chunk_size=250,
        chunk_overlap=25,
    )
    chain.index_documents([Document(page_content="new entry", metadata={"id": "new"})])

    assert source_ids == ["chunk-1", "chunk-2"]
    assert retriever.index_source_calls[0]["source"] == "data/raw/company_handbook.txt"
    assert retriever.index_source_calls[0]["metadata"] == {"category": "docs"}
    assert retriever.index_documents_calls[0][0].page_content == "new entry"

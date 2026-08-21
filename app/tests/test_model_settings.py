from app.config.settings import Settings


def test_settings_reads_model_env_vars(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "env-chat-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "env-embedding-model")
    monkeypatch.setenv("RERANKER_MODEL", "env-reranker-model")

    settings = Settings()

    assert settings.LLM_MODEL == "env-chat-model"
    assert settings.EMBEDDING_MODEL == "env-embedding-model"
    assert settings.RERANKER_MODEL == "env-reranker-model"


def test_bge_embeddings_uses_settings_model(monkeypatch):
    from app.embeddings import bge_embeddings

    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bge_embeddings, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings)
    monkeypatch.setattr(bge_embeddings.settings, "EMBEDDING_MODEL", "env-embedding-model")

    model = bge_embeddings.BGEEmbeddings().get_embedding_model()

    assert captured["model_name"] == "env-embedding-model"
    assert model is not None


def test_groq_llm_uses_settings_model(monkeypatch):
    from app.llm import groq

    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langchain_groq.ChatGroq", FakeChatGroq)
    monkeypatch.setattr(groq.settings, "LLM_MODEL", "env-chat-model")

    llm = groq.get_groq_llm()

    assert captured["model"] == "env-chat-model"
    assert captured["temperature"] == 0
    assert llm is not None

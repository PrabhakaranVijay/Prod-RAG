from app.embeddings.bge_embeddings import BGEEmbeddings


class EmbeddingFactory:

    @staticmethod
    def create_embedding(
        provider: str = "bge"
    ):

        if provider.lower() == "bge":
            return (
                BGEEmbeddings()
                .get_embedding_model()
            )

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )
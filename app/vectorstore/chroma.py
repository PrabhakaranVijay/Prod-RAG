from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import chromadb
from langchain_core.documents import Document

from app.config.constants import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_COLLECTION, TOP_K
from app.config.logging import logger
from app.embeddings.embedding_factory import EmbeddingFactory
from app.ingestion.pipeline import IngestionPipeline


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        embeddings: Any | None = None,
        persist_directory: str | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self.embeddings = embeddings or EmbeddingFactory.create_embedding()

        if client is not None:
            self.client = client
        elif persist_directory is not None:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info("Chroma collection ready: %s", collection_name)

    @staticmethod
    def _document_id(document: Document, index: int) -> str:
        metadata_id = document.metadata.get("id")
        if metadata_id:
            return str(metadata_id)

        payload = f"{index}:{document.page_content}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        if not documents:
            return []

        texts = [document.page_content for document in documents]
        metadatas = [dict(document.metadata or {}) for document in documents]
        embeddings = self.embeddings.embed_documents(texts)
        document_ids = list(ids) if ids is not None else [
            self._document_id(document, index)
            for index, document in enumerate(documents)
        ]

        self.collection.upsert(
            ids=document_ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Indexed %s documents into Chroma.", len(documents))
        return document_ids

    def build_index_from_source(
        self,
        source: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
        chunks = IngestionPipeline.run(
            source=source,
            metadata=metadata or {},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return self.add_documents(chunks)

    def search(self, query: str, top_k: int = TOP_K) -> list[Document]:
        if not query.strip():
            return []

        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        ids = result.get("ids", [[]])[0] or []

        matches: list[Document] = []
        for index, content in enumerate(documents):
            metadata = dict(metadatas[index] or {})
            metadata["id"] = ids[index] if index < len(ids) else metadata.get("id")
            metadata["distance"] = distances[index] if index < len(distances) else None
            metadata["rank"] = index + 1
            matches.append(Document(page_content=content, metadata=metadata))

        return matches

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)


def build_default_vector_store(
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> ChromaVectorStore:
    return ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def index_source(
    source: str,
    metadata: dict[str, Any] | None = None,
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> ChromaVectorStore:
    store = build_default_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    store.build_index_from_source(
        source=source,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return store


def search_source(
    query: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    persist_directory: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = TOP_K,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    store = index_source(
        source=source,
        metadata=metadata,
        persist_directory=persist_directory,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return store.search(query=query, top_k=top_k)


def resolve_source_path(source: str) -> str:
    return str(Path(source).expanduser().resolve())
"""
ChromaDB-backed retriever for RA-ZAD.

Talks to the standalone ChromaDB container (see docker-compose.yml, service
`chromadb`) over HTTP. Embeddings are computed locally via TextEmbedder
(BAAI/bge-base-en-v1.5) rather than relying on ChromaDB's built-in embedding
functions, so we control the query-vs-document instruction-prefix behaviour
described in embedder.py.

This module has no knowledge of which LLM will consume the retrieved
context — that's rag_service.py's job. Swapping the embedding model or the
LLM provider should never require changes here.
"""

from functools import lru_cache

import chromadb

from app.core.config import settings
from app.services.embedder import TextEmbedder, get_embedder


class ChromaRetriever:
    """Retrieves the top-k most similar stored passages for a query."""

    def __init__(self, embedder: TextEmbedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self._client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    def _collection_name(self, name: str) -> str:
        return f"{settings.chroma_collection_prefix}_{name}"

    def get_or_create_collection(self, name: str):
        # No embedding_function passed — we always supply precomputed
        # embeddings ourselves (see embed_documents/embed_query above).
        return self._client.get_or_create_collection(name=self._collection_name(name))

    def collection_exists(self, name: str) -> bool:
        existing = [c.name for c in self._client.list_collections()]
        return self._collection_name(name) in existing

    def index_documents(
        self,
        collection: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> int:
        """Embed and upsert a batch of documents/passages into a collection."""
        coll = self.get_or_create_collection(collection)
        embeddings = self.embedder.embed_documents(texts)
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(ids)

    def retrieve(
        self,
        collection: str,
        query_text: str,
        k: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k passages most similar to query_text.

        Returns a list of {"id", "text", "metadata", "distance"} dicts,
        ordered from most to least similar. Returns [] if the collection
        doesn't exist yet (e.g., before any documents have been indexed).
        """
        if not self.collection_exists(collection):
            return []

        coll = self.get_or_create_collection(collection)
        query_embedding = self.embedder.embed_query(query_text)

        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )

        hits: list[dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0] or [None] * len(ids)
        distances = results.get("distances", [[]])[0] or [None] * len(ids)

        for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return hits


@lru_cache(maxsize=1)
def get_retriever() -> ChromaRetriever:
    """Process-wide singleton so the ChromaDB HTTP client is reused."""
    return ChromaRetriever()

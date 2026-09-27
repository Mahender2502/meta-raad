"""
Text embedding for the RAG pipeline.

Uses BAAI/bge-base-en-v1.5 via sentence-transformers. This is the one piece
of the retrieval stack that's model-specific, so it's isolated here — the
retriever and the RAG service just call .embed_query() / .embed_documents()
and don't care which embedding model backs them.

Note on BGE models: they're trained so that *queries* get a short
instruction prefix ("Represent this sentence for searching relevant
passages: ") while *documents/passages* are embedded as-is, with no prefix.
Using the same encoding for both hurts retrieval quality, so this class
exposes separate methods rather than one generic embed().
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


class TextEmbedder:
    """Thin wrapper around a sentence-transformers model for RAG retrieval."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.query_instruction = (
            query_instruction
            if query_instruction is not None
            else settings.embedding_query_instruction
        )
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages/documents for storage in ChromaDB (no instruction prefix)."""
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (BGE instruction-prefixed)."""
        prefixed = f"{self.query_instruction}{text}"
        vector = self._model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedder:
    """Process-wide singleton so the model is loaded into memory only once."""
    return TextEmbedder()

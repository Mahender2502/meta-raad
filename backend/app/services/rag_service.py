"""
RAG orchestration for anomaly detection.

This is the one place where "retrieval" and "generation" meet, and it is
written to be completely indifferent to which embedding model or which LLM
provider is behind it:

    - Retrieval always goes through ChromaRetriever (app/services/retriever.py).
    - Generation always goes through the LLMClient interface
      (app/services/llm_client.py), obtained via get_llm_client().

Swapping BAAI/bge-base-en-v1.5 for another embedding model, or swapping the
LLM provider from "stub" to "openai"/"deepseek"/anything else, requires no
changes in this file — only in embedder.py / llm_client.py / config.

The actual anomaly-detection prompt templates (Task-Information framing,
JSON output format, reason-before-score ordering, etc. from AD-LLM) will be
added as this evolves; for now this builds a simple, clearly-labelled RAG
prompt so the /rag endpoint is exercisable end-to-end.
"""

from dataclasses import dataclass

from app.services.llm_client import LLMClient, LLMResponse, get_llm_client
from app.services.retriever import ChromaRetriever, get_retriever


@dataclass
class RetrievedContext:
    id: str
    text: str
    metadata: dict
    distance: float | None


@dataclass
class RAGResult:
    query: str
    collection: str
    retrieved: list[RetrievedContext]
    prompt: str
    response: LLMResponse


def build_rag_prompt(query: str, contexts: list[RetrievedContext]) -> str:
    """
    Assemble a prompt from the query plus retrieved exemplars.

    Kept as a small, swappable function — the Meta-RAAD-specific detection
    prompt (normal/anomaly category framing, JSON schema, chain-of-thought
    instructions) will replace or extend this once the detection module is
    built; the retrieval step above it does not change.
    """
    if not contexts:
        context_block = "(no similar examples found in the knowledge base)"
    else:
        context_block = "\n\n".join(
            f"[Example {i + 1}] (similarity distance={c.distance}):\n{c.text}"
            for i, c in enumerate(contexts)
        )

    return (
        "You are given a query and a set of retrieved reference examples. "
        "Use the examples as context to inform your answer about the query.\n\n"
        f"Retrieved examples:\n{context_block}\n\n"
        f"Query:\n{query}\n"
    )


class RAGService:
    def __init__(
        self,
        retriever: ChromaRetriever | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = retriever or get_retriever()
        # Resolved lazily per-call by default (see run()) so the provider
        # configured in settings.llm_provider is honored even if it changes
        # between requests; callers may also inject a specific client.
        self._llm_client = llm_client

    def run(
        self,
        query: str,
        collection: str,
        k: int = 3,
        where: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResult:
        raw_hits = self.retriever.retrieve(collection=collection, query_text=query, k=k, where=where)
        contexts = [
            RetrievedContext(
                id=hit["id"],
                text=hit["text"],
                metadata=hit["metadata"],
                distance=hit["distance"],
            )
            for hit in raw_hits
        ]

        prompt = build_rag_prompt(query, contexts)

        llm_client = self._llm_client or get_llm_client()
        response = llm_client.generate(prompt, temperature=temperature, max_tokens=max_tokens)

        return RAGResult(
            query=query,
            collection=collection,
            retrieved=contexts,
            prompt=prompt,
            response=response,
        )

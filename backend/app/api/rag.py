"""
RAG endpoints.

POST /rag/query   — retrieve top-k similar passages from ChromaDB for a
                     query, build a prompt from them, and generate a
                     response via whichever LLMClient is currently
                     configured (app/services/llm_client.py).
POST /rag/index    — helper endpoint to embed and upsert documents into a
                     ChromaDB collection, so /rag/query has something to
                     retrieve against. Will likely be replaced by an
                     offline ingestion script once real datasets are wired
                     up, but is handy for now.

This module only talks to RAGService — it has no knowledge of ChromaDB,
sentence-transformers, or any particular LLM provider.
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    IndexDocumentsRequest,
    IndexDocumentsResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedContextResponse,
)
from app.services.rag_service import RAGService
from app.services.retriever import get_retriever

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    try:
        service = RAGService()
        result = service.run(
            query=request.query,
            collection=request.collection,
            k=request.k,
            where=request.where,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear API error
        raise HTTPException(status_code=502, detail=f"RAG query failed: {exc}") from exc

    return RAGQueryResponse(
        query=result.query,
        collection=result.collection,
        retrieved=[
            RetrievedContextResponse(
                id=c.id, text=c.text, metadata=c.metadata, distance=c.distance
            )
            for c in result.retrieved
        ],
        prompt=result.prompt,
        response_text=result.response.text,
        llm_model=result.response.model,
    )


@router.post("/index", response_model=IndexDocumentsResponse)
def rag_index(request: IndexDocumentsRequest) -> IndexDocumentsResponse:
    if len(request.ids) != len(request.texts):
        raise HTTPException(status_code=400, detail="ids and texts must be the same length")
    if request.metadatas is not None and len(request.metadatas) != len(request.ids):
        raise HTTPException(status_code=400, detail="metadatas must match ids length if provided")

    try:
        retriever = get_retriever()
        count = retriever.index_documents(
            collection=request.collection,
            ids=request.ids,
            texts=request.texts,
            metadatas=request.metadatas,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear API error
        raise HTTPException(status_code=502, detail=f"Indexing failed: {exc}") from exc

    return IndexDocumentsResponse(collection=request.collection, indexed_count=count)

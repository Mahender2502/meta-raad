"""Pydantic request/response models for the API layer."""

from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="The text to run retrieval + generation against.")
    collection: str = Field(
        ...,
        description="Logical name of the ChromaDB collection to search "
        "(e.g. a dataset name like 'ag_news').",
    )
    k: int = Field(3, ge=1, le=20, description="Number of similar examples to retrieve.")
    where: dict | None = Field(
        None, description="Optional ChromaDB metadata filter, e.g. {'category': 'sports'}."
    )
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1)


class RetrievedContextResponse(BaseModel):
    id: str
    text: str
    metadata: dict
    distance: float | None


class RAGQueryResponse(BaseModel):
    query: str
    collection: str
    retrieved: list[RetrievedContextResponse]
    prompt: str
    response_text: str
    llm_model: str


class IndexDocumentsRequest(BaseModel):
    collection: str = Field(..., description="Logical ChromaDB collection name to index into.")
    ids: list[str]
    texts: list[str]
    metadatas: list[dict] | None = None


class IndexDocumentsResponse(BaseModel):
    collection: str
    indexed_count: int

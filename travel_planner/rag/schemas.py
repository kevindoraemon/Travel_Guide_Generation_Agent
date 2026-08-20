"""Data contracts used by ingestion and the hybrid retrieval pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TravelSourceDocument(BaseModel):
    """One travel guide or itinerary before chunking and vectorization."""

    title: str
    content: str
    source_url: str | None = None
    search_engine: str | None = None
    source_type: str = "travel_itinerary"
    language: str = "zh"
    country: str | None = None
    city: str | None = None
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=utc_now)


class QueryRewriteResult(BaseModel):
    rewritten_query: str = Field(description="Standalone retrieval query preserving user intent")


class MetadataConditions(BaseModel):
    """Only exact, payload-filterable conditions belong here."""

    city: str | None = None
    country: str | None = None
    topic: str | None = None
    language: str | None = None
    source_type: str | None = None
    search_engine: str | None = None

    def active(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self.model_dump().items()
            if isinstance(value, str) and value.strip()
        }


class QueryPlan(BaseModel):
    original_query: str
    rewritten_query: str
    filters: MetadataConditions = Field(default_factory=MetadataConditions)


class RetrievedChunk(BaseModel):
    """Final evidence returned to LangGraph and the LLM."""

    chunk_id: str
    document_id: str
    title: str
    content: str
    source_url: str | None = None
    search_engine: str | None = None
    language: str | None = None
    country: str | None = None
    city: str | None = None
    topic: str | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    original_query: str
    rewritten_query: str
    filters: dict[str, str] = Field(default_factory=dict)
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    reranked_count: int = 0
    deduplicated_count: int = 0
    final_count: int = 0
    reranker_fallback: bool = False


class RetrievalResult(BaseModel):
    evidence: list[RetrievedChunk]
    trace: RetrievalTrace

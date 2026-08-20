"""Auditable Recall@K/MRR evaluation for the Qdrant retrieval pipeline."""

from __future__ import annotations

import time
from math import log2
from statistics import fmean
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

from travel_planner.rag.schemas import MetadataConditions


class RetrievalEvalCase(BaseModel):
    case_id: str
    query: str
    category: str = "general"
    difficulty: str = "medium"
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    graded_relevance: dict[str, int] = Field(
        default_factory=dict,
        description="chunk:<id>/document:<id> -> 0..3 human relevance grade",
    )
    expected_answer_points: list[str] = Field(default_factory=list)
    human_label_status: str = "draft"
    filters: MetadataConditions = Field(default_factory=MetadataConditions)

    @model_validator(mode="after")
    def require_labels(self):
        invalid_keys = [
            key for key in self.graded_relevance
            if not (key.startswith("chunk:") or key.startswith("document:"))
        ]
        invalid_grades = [
            grade for grade in self.graded_relevance.values()
            if not isinstance(grade, int) or not 0 <= grade <= 3
        ]
        if invalid_keys:
            raise ValueError(f"invalid graded relevance keys: {invalid_keys}")
        if invalid_grades:
            raise ValueError("graded relevance values must be integers from 0 to 3")
        if (
            not self.relevant_chunk_ids
            and not self.relevant_document_ids
            and not any(grade > 0 for grade in self.graded_relevance.values())
        ):
            raise ValueError("at least one relevant chunk_id or document_id is required")
        return self

    def relevant_keys(self) -> set[str]:
        explicit = {
            *(f"chunk:{value}" for value in self.relevant_chunk_ids),
            *(f"document:{value}" for value in self.relevant_document_ids),
        }
        graded = {key for key, grade in self.graded_relevance.items() if grade > 0}
        return explicit | graded

    def relevance_grade(self, key: str) -> int:
        if key in self.graded_relevance:
            return self.graded_relevance[key]
        return 1 if key in self.relevant_keys() else 0


class RetrievalCaseResult(BaseModel):
    case_id: str
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_document_ids: list[str]
    matched_relevance_keys: list[str]
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    first_relevant_rank: int | None = None
    latency_ms: float
    trace: dict = Field(default_factory=dict)


class RetrievalEvaluationReport(BaseModel):
    cases: int
    k: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate_at_k: float
    average_latency_ms: float
    results: list[RetrievalCaseResult]


def _item_keys(item) -> set[str]:
    return {f"chunk:{item.chunk_id}", f"document:{item.document_id}"}


def _ndcg_at_k(case: RetrievalEvalCase, evidence: list, k: int) -> float:
    gains = [
        max((case.relevance_grade(key) for key in _item_keys(item)), default=0)
        for item in evidence[:k]
    ]
    dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(gains, 1))
    ideal_grades = sorted(
        (case.relevance_grade(key) for key in case.relevant_keys()),
        reverse=True,
    )[:k]
    idcg = sum(
        (2**grade - 1) / log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, 1)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_retriever(
    retriever,
    cases: Iterable[RetrievalEvalCase],
    *,
    k: int = 5,
) -> RetrievalEvaluationReport:
    """Evaluate labeled queries; no metric is emitted without explicit labels."""

    if k <= 0:
        raise ValueError("k must be positive")
    case_results: list[RetrievalCaseResult] = []

    for case in cases:
        started = time.perf_counter()
        filters = case.filters.active()
        result = retriever.search_with_trace(case.query, top_k=k, **filters)
        latency_ms = (time.perf_counter() - started) * 1000
        evidence = result.evidence[:k]
        relevant = case.relevant_keys()
        matched: set[str] = set()
        first_relevant_rank = None
        for rank, item in enumerate(evidence, 1):
            item_matches = _item_keys(item) & relevant
            matched.update(item_matches)
            if item_matches and first_relevant_rank is None:
                first_relevant_rank = rank

        recall = len(matched) / len(relevant)
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        ndcg = _ndcg_at_k(case, evidence, k)
        case_results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                query=case.query,
                retrieved_chunk_ids=[item.chunk_id for item in evidence],
                retrieved_document_ids=[item.document_id for item in evidence],
                matched_relevance_keys=sorted(matched),
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg,
                first_relevant_rank=first_relevant_rank,
                latency_ms=latency_ms,
                trace=result.trace.model_dump(mode="json"),
            )
        )

    if not case_results:
        raise ValueError("evaluation dataset is empty")
    return RetrievalEvaluationReport(
        cases=len(case_results),
        k=k,
        recall_at_k=fmean(item.recall_at_k for item in case_results),
        mrr=fmean(item.reciprocal_rank for item in case_results),
        ndcg_at_k=fmean(item.ndcg_at_k for item in case_results),
        hit_rate_at_k=fmean(1.0 if item.first_relevant_rank else 0.0 for item in case_results),
        average_latency_ms=fmean(item.latency_ms for item in case_results),
        results=case_results,
    )

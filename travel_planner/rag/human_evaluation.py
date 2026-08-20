"""Human annotation contracts and aggregation for travel QA evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

from travel_planner.rag.evaluation import RetrievalEvalCase
from travel_planner.rag.schemas import MetadataConditions


SCORE_DIMENSIONS = (
    "completeness",
    "faithfulness",
    "citation_correctness",
    "practicality",
    "constraint_satisfaction",
    "fluency",
)

SCORE_WEIGHTS = {
    "completeness": 0.20,
    "faithfulness": 0.25,
    "citation_correctness": 0.20,
    "practicality": 0.15,
    "constraint_satisfaction": 0.15,
    "fluency": 0.05,
}


class TravelQAEvalCase(BaseModel):
    """One source-grounded travel QA case before model execution."""

    case_id: str
    query: str
    category: str
    difficulty: str = "medium"
    answerable: bool = True
    expected_answer_points: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    graded_relevance: dict[str, int] = Field(default_factory=dict)
    filters: MetadataConditions = Field(default_factory=MetadataConditions)
    source_snapshot: str
    human_label_status: str = "draft_needs_review"
    labeler_ids: list[str] = Field(default_factory=list)
    adjudicator_id: str | None = None

    @model_validator(mode="after")
    def validate_ground_truth(self):
        invalid_relevance = [
            (key, grade)
            for key, grade in self.graded_relevance.items()
            if not (key.startswith("chunk:") or key.startswith("document:"))
            or not isinstance(grade, int)
            or not 0 <= grade <= 3
        ]
        if invalid_relevance:
            raise ValueError(f"invalid graded relevance labels: {invalid_relevance}")
        if self.answerable:
            if not self.expected_answer_points:
                raise ValueError("answerable case requires expected_answer_points")
            if not (
                self.relevant_chunk_ids
                or self.relevant_document_ids
                or any(grade > 0 for grade in self.graded_relevance.values())
            ):
                raise ValueError("answerable case requires retrieval relevance labels")
        if self.human_label_status == "adjudicated":
            if len(set(self.labeler_ids)) < 2:
                raise ValueError("adjudicated cases require at least two independent labelers")
            if not self.adjudicator_id:
                raise ValueError("adjudicated cases require an adjudicator_id")
        return self

    def to_retrieval_case(self) -> RetrievalEvalCase:
        if not self.answerable:
            raise ValueError("unanswerable QA cases are not retrieval-recall cases")
        return RetrievalEvalCase(
            case_id=self.case_id,
            query=self.query,
            category=self.category,
            difficulty=self.difficulty,
            relevant_chunk_ids=self.relevant_chunk_ids,
            relevant_document_ids=self.relevant_document_ids,
            graded_relevance=self.graded_relevance,
            expected_answer_points=self.expected_answer_points,
            human_label_status=self.human_label_status,
            filters=self.filters,
        )


class HumanAnswerAnnotation(BaseModel):
    assignment_id: str
    annotator_id: str
    case_id: str
    blind_answer_id: str
    completeness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    citation_correctness: int = Field(ge=1, le=5)
    practicality: int = Field(ge=1, le=5)
    constraint_satisfaction: int = Field(ge=1, le=5)
    fluency: int = Field(ge=1, le=5)
    critical_error: bool
    critical_error_type: str = ""
    notes: str = ""

    def composite_score(self) -> float:
        return sum(getattr(self, key) * weight for key, weight in SCORE_WEIGHTS.items())


def load_qa_cases(path: Path) -> list[TravelQAEvalCase]:
    cases = [
        TravelQAEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id in QA dataset")
    return cases


def load_completed_annotations(path: Path) -> list[HumanAnswerAnnotation]:
    annotations: list[HumanAnswerAnnotation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            payload = dict(row)
            raw_critical = str(payload.get("critical_error", "")).strip().lower()
            truthy = {"1", "true", "yes", "是"}
            falsy = {"0", "false", "no", "否"}
            if raw_critical not in truthy | falsy:
                raise ValueError(
                    f"{payload.get('assignment_id')}: critical_error must be yes/no, not blank"
                )
            payload["critical_error"] = raw_critical in truthy
            annotations.append(HumanAnswerAnnotation.model_validate(payload))
    return annotations


def quadratic_weighted_kappa(pairs: Iterable[tuple[int, int]], *, levels: int = 5) -> float:
    pairs = list(pairs)
    if not pairs:
        return 0.0
    observed = [[0.0] * levels for _ in range(levels)]
    left = [0.0] * levels
    right = [0.0] * levels
    for first, second in pairs:
        i, j = first - 1, second - 1
        observed[i][j] += 1
        left[i] += 1
        right[j] += 1
    total = float(len(pairs))
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    denominator = float((levels - 1) ** 2)
    for i in range(levels):
        for j in range(levels):
            weight = ((i - j) ** 2) / denominator
            observed_disagreement += weight * observed[i][j] / total
            expected_disagreement += weight * (left[i] * right[j]) / (total * total)
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def aggregate_annotations(
    annotations: list[HumanAnswerAnnotation],
    blind_key: dict[str, str],
) -> dict:
    """Aggregate double annotations after unblinding system names."""

    grouped: dict[tuple[str, str], list[HumanAnswerAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation.case_id, annotation.blind_answer_id)].append(annotation)

    incomplete = [
        key for key, values in grouped.items()
        if len(values) != 2 or len({v.annotator_id for v in values}) != 2
    ]
    if incomplete:
        raise ValueError(f"each answer needs two independent annotators; incomplete={incomplete[:5]}")

    systems: dict[str, list[dict]] = defaultdict(list)
    agreement_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (case_id, blind_id), values in grouped.items():
        values = sorted(values, key=lambda item: item.annotator_id)[:2]
        system = blind_key[blind_id]
        systems[system].append({
            "case_id": case_id,
            "composite_score": fmean(value.composite_score() for value in values),
            "critical_error": any(value.critical_error for value in values),
            "dimensions": {
                dimension: fmean(getattr(value, dimension) for value in values)
                for dimension in SCORE_DIMENSIONS
            },
        })
        for dimension in SCORE_DIMENSIONS:
            agreement_pairs[dimension].append(
                (getattr(values[0], dimension), getattr(values[1], dimension))
            )

    system_summary = {}
    for system, rows in systems.items():
        system_summary[system] = {
            "answers": len(rows),
            "mean_composite_score": fmean(row["composite_score"] for row in rows),
            "critical_error_rate": fmean(1.0 if row["critical_error"] else 0.0 for row in rows),
            "mean_dimensions": {
                dimension: fmean(row["dimensions"][dimension] for row in rows)
                for dimension in SCORE_DIMENSIONS
            },
        }

    paired_comparisons = {}
    for left, right in combinations(sorted(systems), 2):
        left_by_case = {row["case_id"]: row["composite_score"] for row in systems[left]}
        right_by_case = {row["case_id"]: row["composite_score"] for row in systems[right]}
        shared = sorted(set(left_by_case) & set(right_by_case))
        wins = sum(left_by_case[case_id] > right_by_case[case_id] + 0.05 for case_id in shared)
        losses = sum(right_by_case[case_id] > left_by_case[case_id] + 0.05 for case_id in shared)
        paired_comparisons[f"{left}_vs_{right}"] = {
            "paired_cases": len(shared),
            f"{left}_wins": wins,
            "ties": len(shared) - wins - losses,
            f"{right}_wins": losses,
        }

    return {
        "systems": system_summary,
        "paired_comparisons": paired_comparisons,
        "inter_annotator_agreement": {
            dimension: quadratic_weighted_kappa(pairs)
            for dimension, pairs in agreement_pairs.items()
        },
    }


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

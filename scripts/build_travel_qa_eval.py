#!/usr/bin/env python3
"""Build a validated travel-QA JSONL dataset from the adjudicated authoring CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.human_evaluation import TravelQAEvalCase


def parse_json(row: dict, key: str, expected_type):
    raw = (row.get(key) or "").strip()
    if not raw:
        return expected_type()
    value = json.loads(raw)
    if not isinstance(value, expected_type):
        raise ValueError(f"{key} must contain JSON {expected_type.__name__}")
    return value


def parse_bool(value: str) -> bool:
    normalized = (value or "true").strip().lower()
    if normalized in {"1", "true", "yes", "是"}:
        return True
    if normalized in {"0", "false", "no", "否"}:
        return False
    raise ValueError(f"answerable must be true/false, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("authoring_csv", type=Path)
    parser.add_argument("--snapshot", required=True, help="Frozen corpus version/hash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()

    cases: list[TravelQAEvalCase] = []
    with args.authoring_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            query = (row.get("query") or "").strip()
            if not query:
                continue
            labelers = [
                value.strip()
                for value in (row.get("labeler_1", ""), row.get("labeler_2", ""))
                if value.strip()
            ]
            adjudicator = (row.get("adjudicator_id") or "").strip() or None
            label_1 = parse_json(row, "relevance_label_1_json", dict)
            label_2 = parse_json(row, "relevance_label_2_json", dict)
            adjudicated = parse_json(row, "adjudicated_relevance_json", dict)
            status = (
                "adjudicated"
                if len(set(labelers)) >= 2 and adjudicator and label_1 and label_2 and adjudicated
                else "draft_needs_review"
            )
            if status != "adjudicated" and not args.allow_draft:
                raise ValueError(
                    f"line {line_number}: two labelers plus adjudicator are required; "
                    "use --allow-draft only for a non-reportable pilot"
                )
            relevant_chunk_ids = [
                key.removeprefix("chunk:")
                for key, grade in adjudicated.items()
                if key.startswith("chunk:") and grade > 0
            ]
            relevant_document_ids = [
                key.removeprefix("document:")
                for key, grade in adjudicated.items()
                if key.startswith("document:") and grade > 0
            ]
            case = TravelQAEvalCase(
                case_id=(row.get("draft_case_id") or "").strip(),
                query=query,
                category=(row.get("category") or "general").strip(),
                difficulty=(row.get("difficulty") or "medium").strip(),
                answerable=parse_bool(row.get("answerable", "true")),
                expected_answer_points=parse_json(row, "expected_answer_points_json", list),
                hard_constraints=parse_json(row, "hard_constraints_json", list),
                relevant_chunk_ids=relevant_chunk_ids,
                relevant_document_ids=relevant_document_ids,
                graded_relevance=adjudicated,
                filters={"city": (row.get("city") or "").strip() or None},
                source_snapshot=args.snapshot,
                human_label_status=status,
                labeler_ids=labelers,
                adjudicator_id=adjudicator,
            )
            cases.append(case)
    if not cases:
        raise SystemExit("no completed question rows found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json(exclude_none=True) + "\n")
    print(json.dumps({
        "cases": len(cases),
        "status": {
            status: sum(case.human_label_status == status for case in cases)
            for status in sorted({case.human_label_status for case in cases})
        },
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

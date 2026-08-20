#!/usr/bin/env python3
"""Blind one or more system answer files and create double-annotation CSV tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.human_evaluation import SCORE_DIMENSIONS, dump_json, load_qa_cases


def parse_named_file(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SYSTEM=answers.jsonl")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("system name cannot be empty")
    return name.strip(), Path(raw_path)


def load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = row.get("answer") or row.get("final_answer") or row.get("final_itinerary")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{path}: case {row.get('case_id')} has no answer text")
        answers[str(row["case_id"])] = answer
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--answers", action="append", type=parse_named_file, required=True)
    parser.add_argument("--annotators", default="annotator_a,annotator_b")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    annotators = [value.strip() for value in args.annotators.split(",") if value.strip()]
    if len(set(annotators)) < 2:
        raise SystemExit("at least two distinct annotators are required")
    cases = load_qa_cases(args.dataset)
    case_by_id = {case.case_id: case for case in cases}
    systems = [(name, load_answers(path)) for name, path in args.answers]
    for name, answers in systems:
        missing = sorted(set(case_by_id) - set(answers))
        if missing:
            raise SystemExit(f"{name} is missing answers for {missing[:10]}")

    rng = random.Random(args.seed)
    blinded: list[dict] = []
    blind_key: dict[str, str] = {}
    for case in cases:
        rows = []
        for system_name, answers in systems:
            digest = hashlib.sha256(
                f"{args.seed}\x1f{case.case_id}\x1f{system_name}".encode("utf-8")
            ).hexdigest()[:12]
            blind_id = f"answer-{digest}"
            blind_key[blind_id] = system_name
            rows.append({"blind_answer_id": blind_id, "answer": answers[case.case_id]})
        rng.shuffle(rows)
        for position, row in enumerate(rows, 1):
            blinded.append({"case": case, "position": position, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "blind_key.json", blind_key)
    fields = [
        "assignment_id", "annotator_id", "case_id", "blind_answer_id", "position",
        "query", "expected_answer_points", "hard_constraints", "answer",
        *SCORE_DIMENSIONS, "critical_error", "critical_error_type", "notes",
    ]
    task_path = args.output_dir / "answer_review.csv"
    with task_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for annotator in annotators:
            rows = list(blinded)
            rng.shuffle(rows)
            for index, item in enumerate(rows, 1):
                case = item["case"]
                writer.writerow({
                    "assignment_id": f"{annotator}-{index:05d}",
                    "annotator_id": annotator,
                    "case_id": case.case_id,
                    "blind_answer_id": item["blind_answer_id"],
                    "position": item["position"],
                    "query": case.query,
                    "expected_answer_points": json.dumps(case.expected_answer_points, ensure_ascii=False),
                    "hard_constraints": json.dumps(case.hard_constraints, ensure_ascii=False),
                    "answer": item["answer"],
                    **{dimension: "" for dimension in SCORE_DIMENSIONS},
                    "critical_error": "",
                    "critical_error_type": "",
                    "notes": "",
                })

    print(json.dumps({
        "cases": len(cases),
        "systems": [name for name, _ in systems],
        "annotators": annotators,
        "annotation_rows": len(blinded) * len(annotators),
        "review_sheet": str(task_path.resolve()),
        "blind_key": str((args.output_dir / "blind_key.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Validate and aggregate completed blind travel-QA human annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.human_evaluation import (
    aggregate_annotations,
    dump_json,
    load_completed_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("blind_key", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = load_completed_annotations(args.review_csv)
    key = json.loads(args.blind_key.read_text(encoding="utf-8"))
    report = aggregate_annotations(annotations, key)
    dump_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

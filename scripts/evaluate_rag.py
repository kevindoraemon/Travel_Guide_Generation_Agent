#!/usr/bin/env python3
"""Evaluate Qdrant RAG with Recall/MRR/nDCG on a labeled JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.evaluation import RetrievalEvalCase, evaluate_retriever


def load_cases(path: Path) -> tuple[list[RetrievalEvalCase], str, int]:
    raw = path.read_bytes()
    cases = []
    excluded_unanswerable = 0
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("answerable") is False:
            excluded_unanswerable += 1
            continue
        cases.append(RetrievalEvalCase.model_validate(payload))
    return cases, hashlib.sha256(raw).hexdigest(), excluded_unanswerable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--disable-query-llm",
        action="store_true",
        help="Use deterministic query rewrite and metadata extraction for offline evaluation.",
    )
    args = parser.parse_args()
    if not 1 <= args.k <= 5:
        raise SystemExit("--k must be between 1 and 5 because the production pipeline returns Top 5")
    if args.disable_query_llm:
        os.environ["RAG_QUERY_REWRITE_ENABLED"] = "false"
        os.environ["RAG_METADATA_EXTRACTION_ENABLED"] = "false"

    from travel_planner.rag import get_travel_retriever

    cases, dataset_sha256, excluded_unanswerable = load_cases(args.dataset)
    retriever = get_travel_retriever()
    try:
        report = evaluate_retriever(retriever, cases, k=args.k)
    finally:
        retriever.close()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "query_llm_enabled": not args.disable_query_llm,
        "excluded_unanswerable_cases": excluded_unanswerable,
        **report.model_dump(mode="json"),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

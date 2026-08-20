#!/usr/bin/env python3
"""Validate travel QA labels and optionally check IDs against local Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.config import get_rag_config
from travel_planner.rag.human_evaluation import load_qa_cases


def corpus_ids() -> tuple[set[str], set[str]]:
    from qdrant_client import QdrantClient

    config = get_rag_config()
    client = QdrantClient(path=config.qdrant_path)
    chunks: set[str] = set()
    documents: set[str] = set()
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                config.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunks.add(str(payload.get("chunk_id")))
                documents.add(str(payload.get("document_id")))
            if offset is None:
                break
    finally:
        client.close()
    return chunks, documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--check-qdrant", action="store_true")
    parser.add_argument("--require-adjudicated", action="store_true")
    args = parser.parse_args()

    cases = load_qa_cases(args.dataset)
    errors: list[str] = []
    if args.require_adjudicated:
        errors.extend(
            f"{case.case_id}: status={case.human_label_status}"
            for case in cases
            if case.human_label_status != "adjudicated"
        )
    if args.check_qdrant:
        chunks, documents = corpus_ids()
        for case in cases:
            missing_chunks = sorted(set(case.relevant_chunk_ids) - chunks)
            missing_documents = sorted(set(case.relevant_document_ids) - documents)
            if missing_chunks or missing_documents:
                errors.append(
                    f"{case.case_id}: missing chunks={missing_chunks}, documents={missing_documents}"
                )

    payload = {
        "dataset": str(args.dataset.resolve()),
        "cases": len(cases),
        "categories": Counter(case.category for case in cases),
        "difficulties": Counter(case.difficulty for case in cases),
        "label_status": Counter(case.human_label_status for case in cases),
        "answerable": Counter(str(case.answerable).lower() for case in cases),
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=dict))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


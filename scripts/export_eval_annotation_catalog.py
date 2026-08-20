#!/usr/bin/env python3
"""Export Qdrant chunks and a question-authoring sheet for human eval construction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.config import get_rag_config


def scroll_all(client, collection: str) -> list[dict]:
    records: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(dict(point.payload or {}) for point in points)
        if offset is None:
            return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/annotation/corpus_v1"))
    parser.add_argument("--questions-per-document", type=int, default=2)
    args = parser.parse_args()
    if args.questions_per_document < 1:
        raise SystemExit("--questions-per-document must be positive")

    from qdrant_client import QdrantClient

    config = get_rag_config()
    client = QdrantClient(path=config.qdrant_path)
    try:
        records = scroll_all(client, config.collection_name)
    finally:
        client.close()
    if not records:
        raise SystemExit("Qdrant collection is empty; ingest the travel corpus first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output_dir / "chunk_catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: (item.get("document_id", ""), item.get("chunk_index", 0))):
            payload = {
                key: record.get(key)
                for key in (
                    "chunk_id", "document_id", "chunk_index", "title", "content", "source_url",
                    "city", "country", "topic", "language", "search_engine", "updated_at",
                )
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    documents: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        documents[str(record.get("document_id"))].append(record)

    authoring_path = args.output_dir / "question_authoring.csv"
    fields = [
        "draft_case_id", "source_document_id", "source_chunk_ids", "title", "city",
        "query", "category", "difficulty", "answerable", "expected_answer_points_json",
        "hard_constraints_json", "relevance_label_1_json", "relevance_label_2_json",
        "adjudicated_relevance_json", "labeler_1", "labeler_2", "adjudicator_id", "notes",
    ]
    with authoring_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for document_index, (document_id, chunks) in enumerate(sorted(documents.items()), 1):
            chunks = sorted(chunks, key=lambda item: item.get("chunk_index", 0))
            chunk_ids = [str(chunk.get("chunk_id")) for chunk in chunks]
            for question_index in range(1, args.questions_per_document + 1):
                writer.writerow({
                    "draft_case_id": f"draft-{document_index:03d}-{question_index:02d}",
                    "source_document_id": document_id,
                    "source_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
                    "title": chunks[0].get("title", ""),
                    "city": chunks[0].get("city", ""),
                    "query": "",
                    "category": "",
                    "difficulty": "medium",
                    "answerable": "true",
                    "expected_answer_points_json": "[]",
                    "hard_constraints_json": "[]",
                    "relevance_label_1_json": "{}",
                    "relevance_label_2_json": "{}",
                    "adjudicated_relevance_json": "{}",
                    "labeler_1": "",
                    "labeler_2": "",
                    "adjudicator_id": "",
                    "notes": "",
                })

    print(json.dumps({
        "collection": config.collection_name,
        "documents": len(documents),
        "chunks": len(records),
        "catalog": str(catalog_path.resolve()),
        "authoring_sheet": str(authoring_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

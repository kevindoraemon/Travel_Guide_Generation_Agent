"""Run the local Qdrant hybrid retrieval pipeline and print its trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag import get_travel_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--city")
    parser.add_argument("--country")
    parser.add_argument("--topic")
    parser.add_argument("--language")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = get_travel_retriever()
    try:
        result = retriever.search_with_trace(
            args.query,
            city=args.city,
            country=args.country,
            topic=args.topic,
            language=args.language,
            top_k=args.top_k,
        )
    finally:
        retriever.close()
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Chunk and index JSONL, Markdown, TXT or HTML travel data in local Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from travel_planner.rag.ingestion import TravelKnowledgeIngestor
from travel_planner.rag.schemas import TravelSourceDocument


SUPPORTED_SUFFIXES = {".jsonl", ".md", ".txt", ".html", ".htm"}


def _html_to_text(value: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(value, "html.parser")
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def _record_to_document(record: dict, *, source_path: Path) -> TravelSourceDocument:
    metadata = dict(record.get("metadata") or {})
    metadata.setdefault("source_file", str(source_path.resolve()))
    content = record.get("content") or record.get("text") or record.get("body") or ""
    values = dict(
        title=record.get("title") or source_path.stem,
        content=content,
        source_url=record.get("source_url") or record.get("url"),
        search_engine=record.get("search_engine"),
        source_type=record.get("source_type", "travel_itinerary"),
        language=record.get("language", "zh"),
        country=record.get("country"),
        city=record.get("city"),
        topic=record.get("topic"),
        metadata=metadata,
    )
    fetched_at = record.get("fetched_at") or record.get("crawled_at")
    if fetched_at:
        values["fetched_at"] = fetched_at
    return TravelSourceDocument(**values)


def load_documents(path: Path) -> list[TravelSourceDocument]:
    files = [path] if path.is_file() else [
        item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    documents: list[TravelSourceDocument] = []

    for source_file in sorted(files):
        suffix = source_file.suffix.lower()
        raw = source_file.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            for line_number, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("metadata", {})["source_line"] = line_number
                documents.append(_record_to_document(record, source_path=source_file))
            continue

        content = _html_to_text(raw) if suffix in {".html", ".htm"} else raw
        documents.append(TravelSourceDocument(
            title=source_file.stem,
            content=content,
            metadata={"source_file": str(source_file.resolve())},
        ))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL 文件或资料目录")
    parser.add_argument("--setup-only", action="store_true", help="只检查连接并创建索引")
    args = parser.parse_args()

    ingestor = TravelKnowledgeIngestor()
    try:
        ingestor.setup()
        if args.setup_only:
            print(json.dumps({"status": "ok", "database": ingestor.store.stats()}, ensure_ascii=False))
            return

        documents = load_documents(args.input)
        if not documents:
            raise SystemExit(f"没有在 {args.input} 找到可入库资料")
        result = ingestor.ingest(documents)
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    finally:
        ingestor.close()


if __name__ == "__main__":
    main()

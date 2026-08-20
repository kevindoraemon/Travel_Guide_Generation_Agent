"""Travel document ingestion into the local Qdrant collection."""

from __future__ import annotations

from travel_planner.rag.config import get_rag_config
from travel_planner.rag.qdrant_store import QdrantTravelStore
from travel_planner.rag.schemas import TravelSourceDocument


class TravelKnowledgeIngestor:
    def __init__(self, store: QdrantTravelStore | None = None):
        self.store = store or QdrantTravelStore(get_rag_config())

    def setup(self) -> None:
        self.store.ping()
        self.store.ensure_collection()

    def close(self) -> None:
        self.store.close()

    def ingest(self, documents: list[TravelSourceDocument]) -> dict:
        results = [self.store.upsert_document(document) for document in documents]
        return {
            "ingested_documents": len(results),
            "ingested_chunks": sum(item["chunk_count"] for item in results),
            "database": self.store.stats(),
            "results": results,
        }

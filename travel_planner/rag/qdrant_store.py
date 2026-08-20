"""Persistent local Qdrant store with named dense and sparse vectors."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from travel_planner.rag.chunking import chunk_text, normalize_text
from travel_planner.rag.config import RagConfig
from travel_planner.rag.embeddings import LocalHybridEmbeddings
from travel_planner.rag.query_processing import infer_topics
from travel_planner.rag.schemas import TravelSourceDocument, utc_now


_POINT_NAMESPACE = uuid.UUID("40a84128-1960-4bbf-8752-480ef2dd7447")


def _canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def _stable_uuid(*values: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, "\x1f".join(values)))


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class QdrantTravelStore:
    """Store chunks and execute the two independent retrieval branches."""

    def __init__(
        self,
        config: RagConfig,
        *,
        client=None,
        embeddings: LocalHybridEmbeddings | None = None,
    ):
        self.config = config
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError(
                    "qdrant-client is required; install project requirements first"
                ) from exc
            client = QdrantClient(path=config.qdrant_path)
        self.client = client
        self.embeddings = embeddings or LocalHybridEmbeddings(config)

    @staticmethod
    def _models():
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError("qdrant-client is required") from exc
        return models

    def ensure_collection(self) -> None:
        models = self._models()
        if self.client.collection_exists(self.config.collection_name):
            return
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                self.config.dense_vector_name: models.VectorParams(
                    size=self.config.dense_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def ping(self) -> None:
        # Local mode is in-process; listing collections exercises the storage path.
        self.client.get_collections()

    def close(self) -> None:
        self.client.close()

    def upsert_document(self, source: TravelSourceDocument) -> dict[str, Any]:
        models = self._models()
        content = normalize_text(source.content)
        if not content:
            raise ValueError("document content is empty")

        canonical_url = _canonical_url(source.source_url)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        document_identity = canonical_url or f"{source.title}\x1f{content_hash}"
        document_id = _stable_uuid("document", document_identity)
        now = utc_now().isoformat()
        inferred_topics = infer_topics(f"{source.title}\n{content}")
        topics = list(dict.fromkeys([value for value in [source.topic, *inferred_topics] if value]))
        primary_topic = source.topic or (topics[0] if topics else None)
        pieces = chunk_text(
            content,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        vector_texts = [f"{source.title}\n{piece}" for piece in pieces]
        dense_vectors = self.embeddings.embed_passages(vector_texts)

        points = []
        for index, (piece, vector_text, dense_vector) in enumerate(
            zip(pieces, vector_texts, dense_vectors, strict=True)
        ):
            chunk_id = _stable_uuid("chunk", document_id, str(index), piece)
            sparse_indices, sparse_values = self.embeddings.embed_sparse(vector_text)
            payload = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": index,
                "title": source.title,
                "content": piece,
                "source_url": canonical_url,
                "search_engine": source.search_engine,
                "source_type": source.source_type,
                "language": source.language,
                "country": source.country,
                "city": source.city,
                "topic": primary_topic,
                "topics": topics,
                "metadata": {key: _json_value(value) for key, value in source.metadata.items()},
                "fetched_at": source.fetched_at.isoformat(),
                "document_content_hash": content_hash,
                "chunk_content_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                "updated_at": now,
            }
            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector={
                        self.config.dense_vector_name: dense_vector,
                        self.config.sparse_vector_name: models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload=payload,
                )
            )

        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        )
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=selector,
            wait=True,
        )
        if points:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )
        return {
            "document_id": document_id,
            "chunk_count": len(points),
            "content_hash": content_hash,
        }

    def _query_filter(self, filters: dict[str, str] | None):
        if not filters:
            return None
        models = self._models()
        allowed = {"city", "country", "topic", "language", "source_type", "search_engine"}
        conditions = [
            models.FieldCondition(
                key="topics" if key == "topic" else key,
                match=models.MatchValue(value=value),
            )
            for key, value in filters.items()
            if key in allowed and value
        ]
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _point_to_candidate(point, score_key: str) -> dict[str, Any]:
        payload = dict(point.payload or {})
        payload[score_key] = float(point.score)
        payload.setdefault("metadata", {})
        return payload

    def search_dense(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query_vector = self.embeddings.embed_query(query)
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            using=self.config.dense_vector_name,
            query_filter=self._query_filter(filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_candidate(point, "dense_score") for point in response.points]

    def search_sparse(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        models = self._models()
        indices, values = self.embeddings.embed_sparse(query, query=True)
        if not indices:
            return []
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=models.SparseVector(indices=indices, values=values),
            using=self.config.sparse_vector_name,
            query_filter=self._query_filter(filters),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._point_to_candidate(point, "sparse_score") for point in response.points]

    def stats(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.config.collection_name):
            return {"collection": self.config.collection_name, "points": 0}
        result = self.client.count(
            collection_name=self.config.collection_name,
            exact=True,
        )
        return {"collection": self.config.collection_name, "points": int(result.count)}

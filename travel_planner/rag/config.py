"""Configuration for the local Qdrant hybrid RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from travel_planner.utils import load_config


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RagConfig:
    enabled: bool
    qdrant_path: str
    collection_name: str
    dense_vector_name: str
    sparse_vector_name: str
    dense_model: str
    dense_dimension: int
    model_cache_dir: str
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    query_rewrite_enabled: bool
    query_rewrite_role: str
    metadata_extraction_enabled: bool
    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    reranker_top_k: int
    final_k: int
    dedup_similarity_threshold: float
    dedup_by_document: bool
    rag_retrieval_path: str
    reranker_model_path: str
    reranker_device: str
    reranker_dtype: str
    reranker_max_length: int


def get_rag_config(stage: str | None = None) -> RagConfig:
    """Load RAG settings, allowing deployment-sensitive values via env vars."""

    resolved_stage = stage or os.environ.get("STAGE") or "prod"
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    stage_config = load_config(stage_name=resolved_stage, config_path=config_path) or {}
    rag = stage_config.get("rag") or {}
    qdrant = rag.get("qdrant") or {}
    embeddings = rag.get("embeddings") or {}
    chunking = rag.get("chunking") or {}
    query_processing = rag.get("query_processing") or {}
    retrieval = rag.get("retrieval") or {}
    reranker = rag.get("reranker") or {}
    dedup = rag.get("deduplication") or {}

    qdrant_path = os.environ.get("QDRANT_PATH") or qdrant.get("path", "data/qdrant")
    model_cache_dir = os.environ.get("RAG_MODEL_CACHE_DIR") or embeddings.get(
        "cache_dir", "data/models"
    )

    return RagConfig(
        enabled=bool(rag.get("enabled", True)),
        qdrant_path=str(Path(qdrant_path)),
        collection_name=os.environ.get("QDRANT_COLLECTION")
        or qdrant.get("collection", "travel_knowledge"),
        dense_vector_name=qdrant.get("dense_vector_name", "dense"),
        sparse_vector_name=qdrant.get("sparse_vector_name", "sparse"),
        dense_model=os.environ.get("RAG_DENSE_MODEL")
        or embeddings.get("dense_model", "BAAI/bge-small-zh-v1.5"),
        dense_dimension=int(embeddings.get("dense_dimension", 512)),
        model_cache_dir=str(Path(model_cache_dir)),
        embedding_batch_size=int(embeddings.get("batch_size", 16)),
        chunk_size=int(chunking.get("chunk_size", 800)),
        chunk_overlap=int(chunking.get("chunk_overlap", 120)),
        query_rewrite_enabled=_env_bool(
            "RAG_QUERY_REWRITE_ENABLED", bool(query_processing.get("rewrite_enabled", True))
        ),
        query_rewrite_role=query_processing.get("role", "rag_query"),
        metadata_extraction_enabled=_env_bool(
            "RAG_METADATA_EXTRACTION_ENABLED",
            bool(query_processing.get("metadata_extraction_enabled", True)),
        ),
        dense_top_k=int(retrieval.get("dense_top_k", 20)),
        sparse_top_k=int(retrieval.get("sparse_top_k", 20)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
        reranker_top_k=int(retrieval.get("reranker_top_k", 8)),
        final_k=int(retrieval.get("final_k", 5)),
        dedup_similarity_threshold=float(dedup.get("similarity_threshold", 0.86)),
        dedup_by_document=bool(dedup.get("by_document", True)),
        rag_retrieval_path=os.environ.get("RAG_RETRIEVAL_PATH")
        or reranker.get("library_path", ""),
        reranker_model_path=os.environ.get("RAG_RERANKER_MODEL")
        or reranker.get("model_path", ""),
        reranker_device=os.environ.get("RAG_RERANKER_DEVICE")
        or reranker.get("device", "cpu"),
        reranker_dtype=reranker.get("dtype", "fp32"),
        reranker_max_length=int(reranker.get("max_length", 512)),
    )

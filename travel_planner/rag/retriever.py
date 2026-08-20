"""End-to-end hybrid retrieval pipeline for travel evidence."""

from __future__ import annotations

from travel_planner import logging as tp_logging
from travel_planner.rag.chunking import normalize_text, tokenize_search_text
from travel_planner.rag.config import RagConfig, get_rag_config
from travel_planner.rag.qdrant_store import QdrantTravelStore
from travel_planner.rag.query_processing import QueryProcessor
from travel_planner.rag.reranker import LocalRAGRetrievalReranker
from travel_planner.rag.schemas import RetrievalResult, RetrievalTrace, RetrievedChunk


logger = tp_logging.get_logger(__name__)
_RETRIEVER = None


def reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    *,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse rankings while retaining branch scores and ranks for observability."""

    fused: dict[str, dict] = {}
    for branch, candidates in (("dense", dense), ("sparse", sparse)):
        for rank, candidate in enumerate(candidates, 1):
            chunk_id = str(candidate["chunk_id"])
            item = fused.setdefault(chunk_id, dict(candidate))
            item[f"{branch}_rank"] = rank
            item[f"{branch}_score"] = candidate.get(f"{branch}_score")
            item["rrf_score"] = float(item.get("rrf_score") or 0.0) + 1.0 / (rrf_k + rank)
    return sorted(
        fused.values(),
        key=lambda item: (-float(item.get("rrf_score") or 0.0), str(item.get("chunk_id"))),
    )


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize_search_text(left))
    right_tokens = set(tokenize_search_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def deduplicate_candidates(
    candidates: list[dict],
    *,
    similarity_threshold: float,
    by_document: bool,
) -> list[dict]:
    """Remove same-document, exact and near-duplicate chunks in rank order."""

    selected: list[dict] = []
    document_ids: set[str] = set()
    fingerprints: set[str] = set()

    for candidate in candidates:
        document_id = str(candidate.get("document_id") or "")
        if by_document and document_id and document_id in document_ids:
            continue
        content = normalize_text(str(candidate.get("content") or ""))
        fingerprint = "".join(content.lower().split())
        if not fingerprint or fingerprint in fingerprints:
            continue
        if any(
            _token_jaccard(content, str(existing.get("content") or "")) >= similarity_threshold
            for existing in selected
        ):
            continue
        selected.append(candidate)
        fingerprints.add(fingerprint)
        if document_id:
            document_ids.add(document_id)
    return selected


class TravelKnowledgeRetriever:
    """Query rewrite → filters → dense/sparse → RRF → rerank → dedup → Top 5."""

    def __init__(
        self,
        config: RagConfig,
        *,
        store: QdrantTravelStore | None = None,
        reranker: LocalRAGRetrievalReranker | None = None,
        query_processor: QueryProcessor | None = None,
    ):
        self.config = config
        self.store = store or QdrantTravelStore(config)
        self.reranker = reranker or LocalRAGRetrievalReranker(config)
        self.query_processor = query_processor or QueryProcessor(config)

    def search_with_trace(
        self,
        query: str,
        *,
        city: str | None = None,
        country: str | None = None,
        topic: str | None = None,
        language: str | None = None,
        source_type: str | None = None,
        search_engine: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        resolved_top_k = min(max(1, top_k or self.config.final_k), self.config.final_k)
        plan = self.query_processor.plan(
            query,
            overrides={
                "city": city,
                "country": country,
                "topic": topic,
                "language": language,
                "source_type": source_type,
                "search_engine": search_engine,
            },
        )
        filters = plan.filters.active()
        self.store.ensure_collection()

        dense = self.store.search_dense(
            plan.rewritten_query,
            filters=filters,
            limit=self.config.dense_top_k,
        )
        sparse = self.store.search_sparse(
            plan.rewritten_query,
            filters=filters,
            limit=self.config.sparse_top_k,
        )
        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=self.config.rrf_k)

        reranker_fallback = False
        try:
            reranked = self.reranker.rerank(
                plan.rewritten_query,
                fused,
                top_k=self.config.reranker_top_k,
            )
        except Exception as exc:
            reranker_fallback = True
            logger.warning("RAG reranker unavailable; using RRF order: %s", exc)
            reranked = fused[: self.config.reranker_top_k]

        deduplicated = deduplicate_candidates(
            reranked,
            similarity_threshold=self.config.dedup_similarity_threshold,
            by_document=self.config.dedup_by_document,
        )
        final_candidates = deduplicated[:resolved_top_k]
        evidence = [RetrievedChunk.model_validate(candidate) for candidate in final_candidates]
        trace = RetrievalTrace(
            original_query=query,
            rewritten_query=plan.rewritten_query,
            filters=filters,
            dense_count=len(dense),
            sparse_count=len(sparse),
            fused_count=len(fused),
            reranked_count=len(reranked),
            deduplicated_count=len(deduplicated),
            final_count=len(evidence),
            reranker_fallback=reranker_fallback,
        )
        logger.info(
            "[RAG] dense=%d sparse=%d fused=%d reranked=%d dedup=%d final=%d filters=%s",
            len(dense), len(sparse), len(fused), len(reranked), len(deduplicated), len(evidence), filters,
        )
        return RetrievalResult(evidence=evidence, trace=trace)

    def search(self, query: str, **kwargs) -> list[RetrievedChunk]:
        return self.search_with_trace(query, **kwargs).evidence

    def close(self) -> None:
        self.store.close()


def get_travel_retriever() -> TravelKnowledgeRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        config = get_rag_config()
        if not config.enabled:
            raise RuntimeError("RAG is disabled in config.yml")
        _RETRIEVER = TravelKnowledgeRetriever(config)
    return _RETRIEVER


def clear_travel_retriever_cache() -> None:
    global _RETRIEVER
    _RETRIEVER = None

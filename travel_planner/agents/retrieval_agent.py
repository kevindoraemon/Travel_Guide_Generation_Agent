"""Main-graph node that retrieves local Qdrant evidence before drafting."""

from __future__ import annotations

import asyncio

from travel_planner import logging as tp_logging
from travel_planner.rag.retriever import get_travel_retriever
from travel_planner.states import AgentState


logger = tp_logging.get_logger(__name__)


async def retrieve_base_knowledge(state: AgentState) -> dict:
    trip_brief = state.get("trip_brief", "")
    if not trip_brief:
        return {"rag_evidence": [], "rag_retrieval_trace": {}}

    try:
        retriever = get_travel_retriever()
        result = await asyncio.to_thread(retriever.search_with_trace, trip_brief)
        logger.info("[RAG] Retrieved %d final evidence chunks", len(result.evidence))
        return {
            "rag_evidence": [item.model_dump(mode="json") for item in result.evidence],
            "rag_retrieval_trace": result.trace.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.warning("[RAG] Retrieval unavailable; continuing without knowledge base: %s", exc)
        return {"rag_evidence": [], "rag_retrieval_trace": {"error": str(exc)}}

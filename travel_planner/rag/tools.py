"""Knowledge-base tool exposed to the Scout agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from travel_planner.rag.retriever import get_travel_retriever


@tool(parse_docstring=True)
def search_travel_knowledge(
    query: str,
    city: str = "",
    country: str = "",
    topic: str = "",
    language: str = "",
    top_k: int = 5,
) -> str:
    """Search the local Qdrant travel knowledge base for reusable evidence.

    Args:
        query: Complete standalone travel question or requirement.
        city: Optional exact city payload filter.
        country: Optional exact country payload filter.
        topic: Optional exact topic such as food, hotel, family or transport.
        language: Optional source-language filter, normally zh or en.
        top_k: Requested evidence count; the pipeline caps this at five.

    Returns:
        JSON evidence with source, dense/sparse/RRF and reranker scores.
    """

    chunks = get_travel_retriever().search(
        query,
        city=city or None,
        country=country or None,
        topic=topic or None,
        language=language or None,
        top_k=top_k,
    )
    return json.dumps(
        [chunk.model_dump(mode="json") for chunk in chunks],
        ensure_ascii=False,
    )


_travel_knowledge_search_tool = search_travel_knowledge

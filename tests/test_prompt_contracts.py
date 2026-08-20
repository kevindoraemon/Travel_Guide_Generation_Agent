from travel_planner.agents.scout_agent import base_tools
from travel_planner.prompts import (
    COMPRESS_INTEL_HUMAN_PROMPT,
    FINAL_ITINERARY_PROMPT,
    ITINERARY_PLANNING_PROMPT,
    SCOUT_AGENT_PROMPT,
)


def test_prompt_contracts_are_consistent():
    assert "Qdrant" not in SCOUT_AGENT_PROMPT
    assert "search_travel_knowledge" not in SCOUT_AGENT_PROMPT
    assert all(tool.name != "search_travel_knowledge" for tool in base_tools)
    assert "语言无关" in ITINERARY_PLANNING_PROMPT
    assert "请勿概括或改写" not in COMPRESS_INTEL_HUMAN_PROMPT
    assert "正文仅使用 [编号]" in FINAL_ITINERARY_PROMPT

"""把当前对话和分层记忆整理为出行简报与路书草稿。"""

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from travel_planner import logging as tp_logging
from travel_planner.llm import get_chat_model, safe_structured_output
from travel_planner.memory import MemoryPolicy, memory_store
from travel_planner.prompts import DRAFT_ITINERARY_PROMPT, TRIP_BRIEF_PROMPT
from travel_planner.states import AgentInputState, AgentState, DraftItinerary, TripRequirement
from travel_planner.utils import get_today_str


logger = tp_logging.get_logger(__name__)
briefing_model = get_chat_model("briefing")


def plan_trip_brief(state: AgentState) -> dict:
    """读取当前节点允许看到的记忆层，并生成结构化出行简报。"""
    layers = memory_store.load_layers(
        state.get("user_id"),
        state.get("session_metadata"),
        state.get("messages", []),
    )
    profile_updates = state.get("profile_updates") or {}
    if not isinstance(profile_updates, dict):
        raise TypeError("profile_updates must be a mapping")
    layers["profile"].update(profile_updates)
    prompt = TRIP_BRIEF_PROMPT.format(
        messages=MemoryPolicy.render(layers, "briefing"),
        date=get_today_str(),
    )
    logger.debug("plan_trip_brief prompt_length=%d", len(prompt))
    response = safe_structured_output(
        briefing_model,
        TripRequirement,
        [HumanMessage(content=prompt)],
    )
    return {"trip_brief": response.trip_brief, "memory_layers": layers}


def write_draft_itinerary(state: AgentState) -> dict:
    """根据本轮简报和 writer 可读的记忆生成路书草稿。"""
    trip_brief = state.get("trip_brief", "")
    prompt = DRAFT_ITINERARY_PROMPT.format(
        trip_brief=trip_brief,
        memory_context=MemoryPolicy.render(state.get("memory_layers", {}), "writer"),
        date=get_today_str(),
    )
    response = safe_structured_output(
        briefing_model,
        DraftItinerary,
        [HumanMessage(content=prompt)],
    )
    logger.info("[BRIEFING] 路书草稿已生成，长度=%d", len(response.draft_itinerary))
    return {
        "trip_brief": trip_brief,
        "draft_itinerary": response.draft_itinerary,
        "coordinator_messages": [
            "Here is the draft itinerary: " + response.draft_itinerary,
            trip_brief,
        ],
    }


if __name__ == "__main__":
    builder = StateGraph(AgentState, input_schema=AgentInputState)
    builder.add_node("plan_trip_brief", plan_trip_brief)
    builder.add_node("write_draft_itinerary", write_draft_itinerary)
    builder.add_edge(START, "plan_trip_brief")
    builder.add_edge("plan_trip_brief", "write_draft_itinerary")
    builder.add_edge("write_draft_itinerary", END)
    print(builder.compile().get_graph().draw_ascii())

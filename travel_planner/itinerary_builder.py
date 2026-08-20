#***********************************************
#      Filename: itinerary_builder.py
#   Description: 多智能体 + 自进化 旅游路书生成 Builder
#***********************************************


from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from travel_planner.utils import get_today_str
from travel_planner.states import AgentState, AgentInputState
from travel_planner.prompts import FINAL_ITINERARY_PROMPT
from travel_planner.agents import plan_trip_brief, write_draft_itinerary
from travel_planner.agents import coordinator_agent
from travel_planner.llm import get_chat_model
from travel_planner.memory import memory_store


# ===== Config =====

writer_model = get_chat_model("writer")


# ===== FINAL ITINERARY GENERATION =====

async def final_itinerary_generation(state: AgentState):
    """最终路书的生成：用户出行需求，出行简报，情报笔记，路书草稿 => 最终旅游路书
    """

    # 取出所有的情报笔记
    notes = state.get("intel_notes", [])
    findings = "\n".join(notes)

    status = state.get("completion_status", "degraded")
    score = state.get("final_quality_score")
    issues = state.get("unresolved_issues", []) or ["未获得质量验收状态"]

    if status == "failed":
        final_text = (
            "## 生成状态\n\n本次未能形成可安全执行的旅游路书。\n\n"
            "## 尚未解决的问题\n\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\n\n## 下一步\n\n请补充缺失信息或调整约束后重新生成。"
        )
        return {
            "final_itinerary": final_text,
            "messages": ["最终的旅游路书: " + final_text],
            "session_metadata": {},
            "memory_layers": {},
            "profile_updates": {},
            "skill_results": {},
        }

    # 组装 prompt
    final_itinerary_prompt = FINAL_ITINERARY_PROMPT.format(
        trip_brief=state.get("trip_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_itinerary=state.get("draft_itinerary", ""),
        completion_status=status,
        final_quality_score="未评分" if score is None else f"{score:.1f}/10",
        stop_reason=state.get("stop_reason", "unknown"),
        unresolved_issues="\n".join(f"- {issue}" for issue in issues),
    )

    # 生成最后的路书
    final_itinerary = await writer_model.ainvoke([HumanMessage(content=final_itinerary_prompt)])
    final_text = str(final_itinerary.content)
    if status == "degraded" and not final_text.lstrip().startswith("## 生成状态"):
        final_text = (
            "## 生成状态\n\n本方案在达到最大迭代次数后仍未通过全部质量验收，以下内容仅供受限使用。"
            + (f"当前综合评分为 {score:.1f}/10。" if score is not None else "")
            + "\n\n## 尚未解决的问题\n\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\n\n"
            + final_text
        )

    # 只持久化明确的结构化更新、轻量简报和可复用成果路径；会话元数据不入账本。
    memory_store.save_turn(
        state.get("user_id"),
        summary=state.get("trip_brief", ""),
        profile_updates=state.get("profile_updates"),
        skill_results=state.get("skill_results") if status == "passed" else None,
        event_at=state.get("memory_event_at"),
    )

    return {
        "final_itinerary": final_text,
        "messages": ["最终的旅游路书: " + final_text],
        "session_metadata": {},
        "memory_layers": {},
        "profile_updates": {},
        "skill_results": {},
    }


# ===== 图的构建 =====

# 构建旅游路书生成的 workflow
itinerary_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加节点
itinerary_builder.add_node("plan_trip_brief", plan_trip_brief)
itinerary_builder.add_node("write_draft_itinerary", write_draft_itinerary)
itinerary_builder.add_node("coordinator_subgraph", coordinator_agent)
itinerary_builder.add_node("final_itinerary_generation", final_itinerary_generation)

# 添加边
itinerary_builder.add_edge(START, "plan_trip_brief")
itinerary_builder.add_edge("plan_trip_brief", "write_draft_itinerary")
itinerary_builder.add_edge("write_draft_itinerary", "coordinator_subgraph")
itinerary_builder.add_edge("coordinator_subgraph", "final_itinerary_generation")
itinerary_builder.add_edge("final_itinerary_generation", END)

# 编译 graph
agent = itinerary_builder.compile()

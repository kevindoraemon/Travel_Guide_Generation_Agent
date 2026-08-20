#***********************************************
#      Filename: coordinator.py
#   Description: 旅游规划协调官（Coordinator）
#***********************************************

"""用于协调多个 Scout（情报员）子代理的协调官。该模块实现了一种协调者模式，其中：
1. Coordinator（协调官）协调情报搜集活动并分配任务
2. 多个 Scout 子代理独立地处理特定的旅游调研子主题（景点/交通/住宿/美食/门票/天气/签证等）
3. 结果汇总并压缩，结合"自进化"评分与体验官对抗反馈迭代精修路书
Coordinator 采用并行执行方式来提高效率，同时为每个调研主题保持独立的上下文窗口。
"""


import re

from typing_extensions import Literal
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from travel_planner.llm import get_chat_model
from travel_planner.prompts import CRITICAL_ADDRESS_PROMPT, ITINERARY_PLANNING_PROMPT
from travel_planner.agents.scout_agent import scout_agent
from travel_planner.agents.critic_agent import critic_node
from travel_planner.agents.evaluator_agent import evaluate_itinerary_quality
from travel_planner.states import (
    CoordinatorState,
    GatherIntel,
    ItineraryComplete,
    QualityMetric
)
from travel_planner.utils import gather_with_concurrency, get_today_str, retry_async
from travel_planner.tools import _think_tool, _refine_itinerary_tool
from travel_planner import logging as tp_logging

logger = tp_logging.get_logger(__name__)


# 确保 async 在 jupyter 环境的兼容性
try:
    import nest_asyncio
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not Jupyter
except ImportError:
    pass


def get_intel_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """从协调官消息历史记录中的 ToolMessage 对象提取情报笔记。
    当协调官通过 GatherIntel 工具调用将情报搜集任务委托给子代理时，
    每个子代理都会返回其压缩的情报结果（以 ToolMessage 内容形式）。
    此函数提取所有此类 ToolMessage 内容，以得到合并后的最终情报笔记。

    Args：
        messages：协调官对话历史记录中的消息列表

    Return：
        从 ToolMessage 对象中提取的情报笔记字符串列表
    """
    return [
        str(tool_msg.content)
        for tool_msg in filter_messages(messages, include_types=["tool"])
        if getattr(tool_msg, "name", None) == "GatherIntel"
    ]


_CHINESE_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value in _CHINESE_NUMBERS:
        return _CHINESE_NUMBERS[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + _CHINESE_NUMBERS.get(value[1], 0)
    return None


def validate_itinerary_contract(trip_brief: str, draft: str) -> list[str]:
    """对无需模型判断的最低交付契约做保守校验。"""
    issues: list[str] = []
    if len(draft.strip()) < 300:
        issues.append("路书正文不足300字，尚未形成可执行方案")

    requested_match = re.search(r"(\d{1,2}|[一二三四五六七八九十]{1,2})\s*(?:天|日游|days?)", trip_brief, re.I)
    requested_days = _number(requested_match.group(1)) if requested_match else None
    planned_days: set[int] = set()
    for line in draft.splitlines():
        match = re.search(r"\bday\s*(\d{1,2})\b", line, re.I) or re.search(
            r"第\s*(\d{1,2}|[一二三四五六七八九十]{1,2})\s*[天日]", line
        )
        if match:
            value = _number(match.group(1))
            if value:
                planned_days.add(value)
    if requested_days and requested_days > 1 and len(planned_days) < requested_days:
        issues.append(f"用户要求{requested_days}天，但草稿只识别到{len(planned_days)}天的分日安排")

    if not re.search(r"交通|地铁|公交|步行|驾车|打车|火车|高铁|航班|机场|车程|transport|metro|bus|train|flight", draft, re.I):
        issues.append("缺少交通方式或交通衔接说明")
    if re.search(r"预算|费用|人均|budget|cost|\d+\s*元", trip_brief, re.I) and not re.search(
        r"预算|费用|人均|门票|合计|[￥¥$]|\d+\s*元|budget|cost", draft, re.I
    ):
        issues.append("用户提出了预算要求，但草稿缺少费用说明")
    if not re.search(r"(?im)^#{1,6}\s*(?:参考资料|资料来源|来源(?:列表)?|references?|sources?)\s*$", draft):
        issues.append("缺少来源列表，关键事实不可核验")
    if re.search(r"待调研|待补充|\bTBD\b", draft, re.I):
        issues.append("草稿中仍包含待调研或待补充内容")
    return issues


def assess_completion(state: CoordinatorState, latest_metric: QualityMetric | None = None) -> dict:
    history = state.get("quality_history", [])
    metric = latest_metric or (history[-1] if history else None)
    score = float(metric["score"]) if metric else None
    hard_issues = validate_itinerary_contract(
        state.get("trip_brief", ""), state.get("draft_itinerary", "")
    )
    issues = list(hard_issues)
    if score is None:
        issues.append("尚未完成路书质量评分")
    elif score < min_need_repair_score:
        issues.append(f"综合质量评分{score:.1f}/10，低于{min_need_repair_score:g}分阈值")
    if metric and issues and metric.get("feedback"):
        issues.append("评估反馈：" + str(metric["feedback"]).strip())
    issues = list(dict.fromkeys(issue for issue in issues if issue))
    return {
        "passed": score is not None and score >= min_need_repair_score and not hard_issues,
        "score": score,
        "hard_checks_passed": not hard_issues,
        "issues": issues,
    }


def completion_update(
    state: CoordinatorState,
    stop_reason: str,
    *,
    latest_metric: QualityMetric | None = None,
    extra_issues: list[str] | None = None,
) -> dict:
    assessment = assess_completion(state, latest_metric)
    issues = list(dict.fromkeys([*assessment["issues"], *(extra_issues or [])]))
    if assessment["passed"]:
        status = "passed"
    elif state.get("draft_itinerary", "").strip():
        status = "degraded"
    else:
        status = "failed"
    return {
        "intel_notes": get_intel_from_tool_calls(state.get("coordinator_messages", [])),
        "trip_brief": state.get("trip_brief", ""),
        "completion_status": status,
        "final_quality_score": assessment["score"],
        "hard_checks_passed": assessment["hard_checks_passed"],
        "stop_reason": stop_reason,
        "unresolved_issues": issues,
    }



# ===== CONFIGURATION =====

coordinator_tools = [GatherIntel, ItineraryComplete, _think_tool, _refine_itinerary_tool]
coordinator_model = get_chat_model("coordinator")
coordinator_model_with_tools = coordinator_model.bind_tools(coordinator_tools)


# System constants (最大迭代次数 / 最大并行子代理数)
max_planning_iterations = 15  # Calls to think_tool + GatherIntel + refine_itinerary
max_concurrent_scouts = 3     # 最大并行情报员数
min_need_repair_score = 6.0   # 评估低于这个分数，就要触发协调官修复提醒


# ===== COORDINATOR NODES =====

async def coordinator(state: CoordinatorState) -> Command[Literal["coordinator_tools"]]:
    """分析出行需求简报和当前进展
    功能：
        - 需要搜集哪些旅游情报
        - 是否开展并行情报搜集
        - 路书规划何时完成

    Args：
        state：当前协调官状态，包含 messages 和 progress

    Returns：
        用于跳转到 coordinator_tools 节点并更新状态的命令
    """
    coordinator_messages = state.get("coordinator_messages", [])
    iteration = state.get("planning_iterations", 0)
    logger.info("[COORDINATOR] coordinator invoked (iteration=%d, messages=%d)", iteration, len(coordinator_messages))

    # 组装系统提示词
    system_message = ITINERARY_PLANNING_PROMPT.format(
        date=get_today_str(),
        max_concurrent_scouts=max_concurrent_scouts,
        max_planning_iterations=max_planning_iterations
    )
    messages = [SystemMessage(content=system_message)] + coordinator_messages

    # 动态上下文注入：检查并注入任何未处理的体验官对抗反馈，实现自我纠正机制。
    critiques = state.get("active_critiques", [])
    unaddressed = [c for c in critiques if not c.addressed]
    if unaddressed:
        critique_text = "\n".join([f"- {c.author} says: {c.concern}" for c in unaddressed])
        intervention = SystemMessage(content=CRITICAL_ADDRESS_PROMPT.format(critique_text=critique_text))
        messages.append(intervention)

    # 如果上一次迭代中质量得分较低，则会发出提醒
    if state.get("needs_quality_repair"):
        messages.append(SystemMessage(
            content=f"上一稿路书质量较低（得分低于{min_need_repair_score:g}/10），请继续完善。"
        ))

    remaining_iterations = max_planning_iterations - iteration
    if remaining_iterations <= 3:
        messages.append(SystemMessage(content=(
            f"仅剩{max(remaining_iterations, 1)}轮规划预算。停止重复或泛化搜索，优先针对最低评分维度调用"
            "refine_itinerary；最后一轮必须收口，不得用虚高评分掩盖未解决问题。"
        )))

    # 决策调用哪一个工具
    response = await retry_async(lambda: coordinator_model_with_tools.ainvoke(messages))
    logger.info(
        "coordinator model produced tool_calls=%s num_tool_calls=%d",
        bool(response.tool_calls),
        len(response.tool_calls or []),
    )

    # 跳转到 coordinator_tools
    return Command(
        goto="coordinator_tools",
        update={
            "coordinator_messages": [response],
            "planning_iterations": iteration + 1,
            "needs_quality_repair": False  # 在向协调官发出提醒后，重置修复标志
        }
    )


async def coordinator_tools(state: CoordinatorState) -> Command[Literal["coordinator", "__end__"]]:
    """
    执行协调官决策——继续下一轮情报搜集或结束流程。

    功能：
        - 执行 think_tool 调用以进行思考
        - 并行启动针对不同主题的情报员子代理
        - 汇总情报结果
        - 确定路书规划何时完成

    参数：
        state：包含协调官 messages 和迭代次数

    返回值：
        继续下一轮协调官 / 结束流程
    """
    coordinator_messages = state.get("coordinator_messages", [])
    planning_iterations = state.get("planning_iterations", 0)
    most_recent_message = coordinator_messages[-1]

    # 先检查显式完成请求；运行时验收门不会仅凭模型声明放行。
    exceeded_iterations = planning_iterations >= max_planning_iterations
    no_tool_calls = not most_recent_message.tool_calls
    planning_complete = any(
        tool_call["name"] == "ItineraryComplete"
        for tool_call in most_recent_message.tool_calls
    )

    assessment = assess_completion(state)
    if planning_complete:
        if assessment["passed"]:
            logger.info("[ITINERARY] Completion gate passed.")
            return Command(goto=END, update=completion_update(state, "quality_gate_passed"))
        if exceeded_iterations:
            logger.warning("[ITINERARY] Maximum iterations reached below quality gate.")
            return Command(goto=END, update=completion_update(state, "max_iterations_reached"))
        feedback = "完成请求被拒绝，必须继续修复：" + "；".join(assessment["issues"])
        tool_messages = [
            ToolMessage(
                content=feedback if call["name"] == "ItineraryComplete" else "本轮包含未通过的完成请求，请在下一轮重新调用该工具。",
                name=call["name"],
                tool_call_id=call["id"],
            )
            for call in most_recent_message.tool_calls
        ]
        return Command(
            goto="coordinator",
            update={"coordinator_messages": tool_messages, "needs_quality_repair": True},
        )

    if no_tool_calls:
        if assessment["passed"]:
            return Command(goto=END, update=completion_update(state, "quality_gate_passed"))
        if exceeded_iterations:
            return Command(goto=END, update=completion_update(state, "max_iterations_reached"))
        return Command(
            goto="coordinator",
            update={
                "coordinator_messages": [SystemMessage(content=(
                    "当前结果未通过终止条件，请继续调用工具修复：" + "；".join(assessment["issues"])
                ))],
                "needs_quality_repair": True,
            },
        )

    else:
        # 初始化变量
        tool_messages = []
        all_raw_intel = []
        draft_itinerary = state.get("draft_itinerary", "")
        updates = {}
        next_step = "coordinator"

        # 执行所有的工具调用
        try:
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            gather_intel_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "GatherIntel"
            ]

            refine_itinerary_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "refine_itinerary"
            ]

            logger.info(
                "[COORDINATOR] coordinator_tools executing think=%d gather=%d refine=%d",
                len(think_tool_calls),
                len(gather_intel_calls),
                len(refine_itinerary_calls),
            )

            # 调用 think 工具（在调用其他工具之前，必须拿到反思结果）(synchronous)
            for tool_call in think_tool_calls:
                observation = _think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # 调用 GatherIntel 工具 (asynchronous)
            if gather_intel_calls:
                # 并行启动多个情报员子代理
                scout_factories = [
                    (
                        lambda call=tool_call: scout_agent.ainvoke({
                            "scout_messages": [
                                HumanMessage(content=call["args"]["travel_topic"])
                            ],
                            "travel_topic": call["args"]["travel_topic"]
                        })
                    )
                    for tool_call in gather_intel_calls
                ]

                # 即使模型一次生成超过 3 个 GatherIntel，也严格限制最多 3 个 Scout 同时运行。
                tool_results = await gather_with_concurrency(
                    max_concurrent_scouts,
                    scout_factories,
                )

                # 将情报结果格式化为工具消息
                # 每个情报员子代理都会在 result["compressed_intel"] 中返回压缩后的情报
                # 我们将这些压缩后的情报写入 ToolMessage 的内容，以便
                # 协调官可以通过 get_intel_from_tool_calls() 检索到这些结果
                intel_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_intel", "Error synthesizing travel intel"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, gather_intel_calls)
                ]

                tool_messages.extend(intel_tool_messages)

                # 聚合所有的 raw intel
                all_raw_intel = [
                    "\n".join(result.get("raw_intel", []))
                    for result in tool_results
                ]

            # 开始调用大模型结合已有信息精修路书
            for tool_call in refine_itinerary_calls:
                findings = "\n".join(get_intel_from_tool_calls(state.get("coordinator_messages", [])))

                new_draft = _refine_itinerary_tool.invoke({
                    "trip_brief": state.get("trip_brief", ""),
                    "findings": findings,
                    "draft_itinerary": state.get("draft_itinerary", "")
                })

                # 执行 Critical Step：Self-Evolution 的评估
                eval_result = evaluate_itinerary_quality(
                        trip_brief=state.get("trip_brief", ""),
                        draft_itinerary=new_draft
                )
                logger.info(
                    "[EVALUATOR] feasibility score=%f, budget score=%f, experience score=%f",
                    eval_result.feasibility_score,
                    eval_result.budget_score,
                    eval_result.experience_score
                )
                logger.info(f"[EVALUATOR] scoing reason: {eval_result.reason}")

                # 评估路书质量得分：(可执行性得分+预算合理性得分+体验丰富度得分) / 3
                avg_score = (eval_result.feasibility_score + eval_result.budget_score + eval_result.experience_score) / 3

                # 把质量得分追加到 tool message, 供协调官参考
                tool_messages.append(ToolMessage(
                    content=f"Itinerary Updated.\nQuality Score: {avg_score}/10.\nJudge Feedback: {eval_result.reason}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

                draft_itinerary = new_draft
                updates["draft_itinerary"] = draft_itinerary

                # 记录路书质量评分的记录，如果分数低于 min_need_repair_score，把 repair 标志位置位 true
                latest_metric = QualityMetric(
                    score=avg_score,
                    feedback=eval_result.reason,
                    iteration=state.get("planning_iterations", 0))
                updates["quality_history"] = [latest_metric]

                if avg_score < min_need_repair_score:
                    updates["needs_quality_repair"] = True

                # 跳转到 self-correction 节点 (体验官 Critic)
                next_step = "critic"

            # 更新本次迭代状态信息
            updates["coordinator_messages"] = tool_messages
            updates["raw_intel"] = all_raw_intel

            if exceeded_iterations:
                effective_state = {
                    **state,
                    **updates,
                    "draft_itinerary": draft_itinerary,
                    "coordinator_messages": [
                        *state.get("coordinator_messages", []),
                        *updates.get("coordinator_messages", []),
                    ],
                    "quality_history": [
                        *state.get("quality_history", []),
                        *updates.get("quality_history", []),
                    ],
                }
                return Command(
                    goto=END,
                    update={**updates, **completion_update(
                        effective_state,
                        "max_iterations_reached",
                        latest_metric=updates.get("quality_history", [None])[-1],
                    )},
                )

            return Command(goto=next_step, update=updates)

        except Exception as e:
            logger.exception("Coordinator tool execution failed")
            return Command(
                goto=END,
                update=completion_update(
                    state,
                    "coordinator_error",
                    extra_issues=["协调流程执行异常，部分信息可能未完成验证"],
                ),
            )



# ===== GRAPH CONSTRUCTION =====

coordinator_builder = StateGraph(CoordinatorState)
coordinator_builder.add_node("coordinator", coordinator)
coordinator_builder.add_node("coordinator_tools", coordinator_tools)
coordinator_builder.add_node("critic", critic_node)

coordinator_builder.add_edge(START, "coordinator")
coordinator_builder.add_edge("coordinator", "coordinator_tools")
coordinator_builder.add_edge("critic", "coordinator")

coordinator_agent = coordinator_builder.compile()


if __name__ == "__main__":
    print(coordinator_agent.get_graph().draw_ascii())

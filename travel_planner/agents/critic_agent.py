#***********************************************
#      Filename: critic_agent.py
#   Description: 路书体验官（挑刺 / 对抗）智能体
#***********************************************

from langchain_core.messages import SystemMessage, HumanMessage

from travel_planner.prompts import CRITIC_PROMPT
from travel_planner.llm import get_chat_model
from travel_planner.states import CoordinatorState, Critique
from travel_planner.utils import retry_async
from travel_planner import logging as tp_logging

logger = tp_logging.get_logger(__name__)


# 初始化模型
critic_model = get_chat_model("critic")

# CONSTANTS
MAX_CRITIC = 3           # 体验官最大挑刺次数，防止追求完美，无限循环
MIN_DRAFT_LEN = 50       # 路书草稿字数最少不低于 50, 否则不予评判，直接返回
MIN_CRITIC = 20          # 如果体验官输出很短，则判定没有缺陷（防止模型指令遵循不足输出"PASS"以外的其他字符）


async def critic_node(state: CoordinatorState) -> dict:
    """
    这是一个路书体验官（对抗）智能体，用于找出路书的行程冲突、需求偏离和不完善的地方。
    """

    draft = state.get("draft_itinerary", "")
    trip_brief = state.get("trip_brief", "")
    critique_nums = state.get("critique_nums", 0)

    # 设置最大对抗次数
    if critique_nums >= MAX_CRITIC or not draft or len(draft) < MIN_DRAFT_LEN:
        return {}

    # 组装 prompt
    prompt = CRITIC_PROMPT.format(trip_brief=trip_brief, draft_itinerary=draft)

    # 调用体验官大模型获得对抗建议
    response = await retry_async(lambda: critic_model.ainvoke([HumanMessage(content=prompt)]))
    content = response.content

    # 如果"PASS", 则直接返回
    if "PASS" in content or len(content) < MIN_CRITIC:
        return {}

    # 如果找到路书缺陷，返回 critique
    critique = Critique(
        author="路书体验官",
        concern=content,
        addressed=False
    )
    logger.info(f"[CRITIC] {content}")

    # 返回 active_critiques 实现动态上下文注入，并把该意见作为 System Message 注入到协调官的消息历史中
    return {
        "active_critiques": [critique],
        "critique_nums": critique_nums + 1,
        "coordinator_messages": [
            SystemMessage(content=f"ADVERSARIAL FEEDBACK DETECTED: {content}")
        ]
    }

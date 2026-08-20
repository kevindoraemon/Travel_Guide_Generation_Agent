#***********************************************
#      Filename: evaluator_agent.py
#   Description: 路书评估智能体（自进化评分）
#***********************************************

from langchain_core.messages import SystemMessage, HumanMessage

from travel_planner.llm import get_chat_model, safe_structured_output
from travel_planner.states import EvaluationResult
from travel_planner.prompts import ITINERARY_EVALUATOR_PROMPT


# 初始化 Judge Model
judge_model = get_chat_model("evaluator")


def evaluate_itinerary_quality(trip_brief: str, draft_itinerary: str) -> EvaluationResult:
    """
    此函数实现了"自进化"评分机制。用另一个 LLM 作为评判者，以评估路书草稿相对于原始出行需求简报的质量。
    打分的分数和原因会返回给协调官智能体，驱动迭代修复。
    """

    # 组装 prompt
    eval_prompt = ITINERARY_EVALUATOR_PROMPT.format(
            trip_brief=trip_brief,
            draft_itinerary=draft_itinerary
    )

    # 获取结构化的分数结果（可执行性/预算合理性/体验丰富度 三维度打分）
    # 带 fallback，兼容各类 OpenAI 兼容接口
    return safe_structured_output(judge_model, EvaluationResult, [HumanMessage(content=eval_prompt)])

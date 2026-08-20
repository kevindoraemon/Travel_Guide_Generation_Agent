#***********************************************
#      Filename: eval_result.py
#   Description: 路书评估结果结构化输出（自进化评分）
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):

    # 可执行性得分：0-10分衡量行程节奏、交通衔接、时间安排是否现实可执行
    feasibility_score: int = Field(ge=0, le=10, description="0-10 score on executability of the itinerary")

    # 预算合理性得分：衡量路书费用预估是否清晰合理、与用户预算匹配
    budget_score: int = Field(ge=0, le=10, description="0-10 score on budget reasonableness")

    # 体验丰富度得分：衡量路书是否覆盖需求、内容丰富有深度、含实用避坑信息
    experience_score: int = Field(ge=0, le=10, description="0-10 score on richness of experience")

    # 打分原因，用于改善路书质量
    reason: str = Field(description="Feedback for the coordinator")

#***********************************************
#      Filename: coordinator.py
#   Description: 协调官（Coordinator）智能体的结构化字段定义
#***********************************************

"""
多智能体协调官的 State 定义
本文件定义了多智能体协调官工作流程中使用的 State 对象和 tools 字段定义。
"""

import operator
from typing_extensions import Annotated, Literal, NotRequired, TypedDict, Sequence, List

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from travel_planner.states.critique import Critique
from travel_planner.states.quality import QualityMetric


class CoordinatorState(TypedDict):
    """
    多智能体协调官的 State 定义。
    负责协调协调官和情报员（Scout）之间的工作，跟踪路书规划进展并汇总来自多个子 Agent 的情报。
    """

    coordinator_messages: Annotated[Sequence[BaseMessage], add_messages]   # 协调官信息,用于协调和传递信息
    trip_brief: str                                                        # 指导整体路书规划方向的出行需求简报
    intel_notes: Annotated[list[str], operator.add] = []                   # 已处理和结构化的情报笔记，可用于生成最终路书
    planning_iterations: int = 0                                           # 跟踪规划迭代次数的计数器
    critique_nums: int = 0                                                 # 跟踪体验官挑刺次数的计数器
    raw_intel: Annotated[list[str], operator.add] = []                     # 从子代理情报收集中收集的原始未处理情报
    draft_itinerary: str                                                   # 路书草稿
    active_critiques: Annotated[List[Critique], operator.add]              # 用于存放主动评估的内容
    quality_history: Annotated[List[QualityMetric], operator.add]          # 质量评估的历史记录（自进化曲线）
    needs_quality_repair: bool                                             # 评估员可以设置一个 bool 标志，向协调官发出路书草稿质量低的信号
    completion_status: NotRequired[Literal["passed", "degraded", "failed"]]
    final_quality_score: NotRequired[float | None]
    hard_checks_passed: NotRequired[bool]
    stop_reason: NotRequired[str]
    unresolved_issues: NotRequired[list[str]]

@tool
class GatherIntel(BaseModel):
    """用于将情报搜集任务委派给专业情报员子代理（Scout）的工具。"""
    travel_topic: str = Field(
        description="旅游调研主题。每次委派的任务应该为单一主题，并需详细描述（至少一个段落），例如：景点的门票/开放时间、交通班次、住宿美食、签证天气等。",
    )

@tool
class ItineraryComplete(BaseModel):
    """用于指示路书规划过程已完成的工具。"""
    pass

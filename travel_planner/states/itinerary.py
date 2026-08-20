"""旅游路书主图的输入、运行状态和结构化输出。"""

import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Any, Literal, NotRequired, Optional, Sequence


class AgentInputState(MessagesState):
    """调用方可显式传入的记忆字段；未提供 user_id 时不写长期记忆。"""

    user_id: NotRequired[str]
    session_metadata: NotRequired[dict[str, Any]]
    profile_updates: NotRequired[dict[str, Any]]
    skill_results: NotRequired[dict[str, Any]]
    memory_event_at: NotRequired[str]


class AgentState(AgentInputState):
    """多 Agent 旅游路书生成系统的主状态。"""

    memory_layers: NotRequired[dict[str, Any]]
    trip_brief: Optional[str]
    coordinator_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_intel: Annotated[list[str], operator.add]
    intel_notes: Annotated[list[str], operator.add]
    draft_itinerary: str
    final_itinerary: str
    planning_iterations: NotRequired[int]
    quality_history: Annotated[list[dict[str, Any]], operator.add]
    completion_status: NotRequired[Literal["passed", "degraded", "failed"]]
    final_quality_score: NotRequired[float | None]
    hard_checks_passed: NotRequired[bool]
    stop_reason: NotRequired[str]
    unresolved_issues: NotRequired[list[str]]


class TripRequirement(BaseModel):
    trip_brief: str = Field(description="用于指导后续旅游路书规划的详细出行需求简报。")


class DraftItinerary(BaseModel):
    draft_itinerary: str = Field(description="用于后续调研和修订的旅游路书草稿。")

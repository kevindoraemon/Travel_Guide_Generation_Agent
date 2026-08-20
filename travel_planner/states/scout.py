#***********************************************
#      Filename: scout.py
#   Description: 旅游情报员结构化字段定义
#***********************************************

"""旅游情报员（Scout Agent）的 State 字段定义
本文件定义了用于 Scout Agent 工作流程的 State 对象和结构化字段
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== STATE DEFINITIONS =====

class ScoutState(TypedDict):
    """旅游情报员的 State，包含消息历史记录和元数据。
    此状态跟踪情报员的对话、用于限制工具调用次数的迭代计数
    、正在调研的旅游主题、压缩后的情报以及用于详细分析的原始情报笔记。
    """
    scout_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    travel_topic: str
    compressed_intel: str
    raw_intel: Annotated[List[str], operator.add]

class ScoutOutputState(TypedDict):
    """旅游情报员的输出状态，包含最终的情报结果。
    此状态表示情报搜集过程的最终输出，包含压缩后的情报以及过程中的所有原始笔记。
    """
    compressed_intel: str
    raw_intel: Annotated[List[str], operator.add]
    scout_messages: Annotated[Sequence[BaseMessage], add_messages]


# ===== STRUCTURED OUTPUT SCHEMAS =====

class Summary(BaseModel):
    """用于网页内容摘要的结构化字段"""
    summary: str = Field(description="网页内容的简明摘要")
    key_excerpts: str = Field(description="内容中的重要引文和摘录")

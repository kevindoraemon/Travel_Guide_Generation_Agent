#***********************************************
#      Filename: critique.py
#   Description: 路书挑刺（体验官）格式化输出
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field


class Critique(BaseModel):
    """用于接收来自"路书体验官"或其他质量控制 Agent 的对抗性反馈的结构化模型"""

    # 用于追踪生成批评的 Agent（例如，"路书体验官", "安全检查"），以便于问责。
    author: str

    # 在路书草稿中发现的具体行程冲突、需求偏离或事实错误。
    concern: str

    # 用于追踪批评是否已在后续路书修订中得到解决的标志
    addressed: bool = Field(default=False, description="协调官是否已修复此问题？")

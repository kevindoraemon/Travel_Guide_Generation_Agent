#***********************************************
#      Filename: quality.py
#   Description: 路书质量控制格式化输出
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field

class QualityMetric(TypedDict):
    """一个 TypedDict，用于存储特定迭代中路书草稿质量的快照"""

    # 由我们的自进化评估器计算的质量得分
    score: float

    # 评估器提供的解释得分的文本反馈
    feedback: str

    # 记录此得分的迭代次数，用于跟踪随时间推移的进度（自进化曲线）
    iteration: int

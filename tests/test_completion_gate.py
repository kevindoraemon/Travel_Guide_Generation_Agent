import asyncio

from langchain_core.messages import AIMessage
from langgraph.graph import END

from travel_planner.agents.coordinator import coordinator_tools, validate_itinerary_contract


def _draft() -> str:
    return """# 北京两日游

## 第一天
上午参观故宫，下午步行前往景山，晚上乘地铁返回酒店。预计门票和交通费用200元。

## 第二天
乘公交前往颐和园，下午乘地铁返程。两日预算合计约1500元，并预留机动费用。

## 交通与预算
市内以地铁和公交为主，逐段预留换乘时间。住宿、餐饮、门票和交通均计入预算。

以上安排根据公开信息整理，出发前需要再次核对临时公告。""" + "路线安排兼顾距离、开放时间和休息需求。" * 12 + """

### 来源
[1] 故宫博物院官方网站：https://www.dpm.org.cn/
"""


def _state(score: float, iteration: int = 1) -> dict:
    return {
        "trip_brief": "北京2天旅行，预算2000元",
        "draft_itinerary": _draft(),
        "planning_iterations": iteration,
        "quality_history": [{"score": score, "feedback": "测试反馈", "iteration": iteration}],
        "coordinator_messages": [AIMessage(content="", tool_calls=[{
            "name": "ItineraryComplete", "args": {}, "id": "complete-1", "type": "tool_call"
        }])],
    }


def test_completion_gate_accepts_only_passing_result():
    command = asyncio.run(coordinator_tools(_state(7.0)))
    assert command.goto == END
    assert command.update["completion_status"] == "passed"


def test_completion_gate_rejects_early_low_score():
    command = asyncio.run(coordinator_tools(_state(5.0)))
    assert command.goto == "coordinator"
    assert command.update["needs_quality_repair"] is True


def test_completion_gate_degrades_at_max_iterations():
    command = asyncio.run(coordinator_tools(_state(5.0, iteration=15)))
    assert command.goto == END
    assert command.update["completion_status"] == "degraded"
    assert command.update["stop_reason"] == "max_iterations_reached"
    assert command.update["unresolved_issues"]


def test_hard_contract_detects_missing_days_and_sources():
    issues = validate_itinerary_contract("西安3天，预算3000元", "只有一天安排，乘地铁出行，费用100元。")
    assert any("3天" in issue for issue in issues)
    assert any("来源" in issue for issue in issues)


def test_calendar_day_is_not_mistaken_for_trip_duration():
    issues = validate_itinerary_contract("2026年8月16日出发", _draft())
    assert not any("16天" in issue for issue in issues)

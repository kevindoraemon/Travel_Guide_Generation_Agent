#***********************************************
#      Filename: scout_agent.py
#   Description: 旅游情报员（Scout Agent）
#***********************************************

"""旅游情报员核心实现
该文件实现了一个 Scout Agent，它可以执行迭代式网络搜索和综合分析，
为用户的旅游调研主题搜集景点、交通、住宿、美食、门票、天气、签证等情报。
"""


import asyncio
import json

from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.tools import BaseTool

from travel_planner.llm import get_chat_model
from travel_planner.states import ScoutState, ScoutOutputState
from travel_planner.utils import get_today_str, retry_async
from travel_planner.tools import (
    _tavily_search_tool,
    _think_tool,
    get_travel_mcp_tools,
)
from travel_planner.prompts import SCOUT_AGENT_PROMPT, COMPRESS_INTEL_SYSTEM_PROMPT, COMPRESS_INTEL_HUMAN_PROMPT
from travel_planner import logging as tp_logging

logger = tp_logging.get_logger(__name__)


# ===== CONFIGURATION =====

# 本地工具立即可用；MCP 工具在首次执行 Scout 时异步加载。
base_tools = [_tavily_search_tool, _think_tool]
_scout_tools_cache: list[BaseTool] | None = None
MAX_TOOL_CALL_ITERATIONS = 5

# 初始化模型
model = get_chat_model("scout_main")
compress_model = get_chat_model("scout_compressor")


async def get_scout_tools() -> list[BaseTool]:
    """返回 Scout 可用工具；MCP 连接失败时降级为本地搜索工具。"""

    global _scout_tools_cache
    if _scout_tools_cache is not None:
        return _scout_tools_cache

    try:
        mcp_tools = await get_travel_mcp_tools()
    except Exception as exc:
        logger.warning(
            "Travel MCP tools are unavailable; Scout will use Tavily only: %s",
            exc,
        )
        mcp_tools = []

    _scout_tools_cache = [*base_tools, *mcp_tools]
    return _scout_tools_cache


# ===== AGENT NODES =====

async def llm_call(state: ScoutState):
    """根据当前状态决策下一步的动作"""

    msg_count = len(state.get("scout_messages", []))
    logger.debug("llm_call invoked with %d messages", msg_count)

    # MCP 工具 schema 在运行时加载并与本地工具一起绑定到模型。
    runtime_tools = await get_scout_tools()
    model_with_tools = model.bind_tools(runtime_tools)
    response = await retry_async(lambda: model_with_tools.ainvoke(
        [SystemMessage(content=SCOUT_AGENT_PROMPT.format(date=get_today_str()))]
        + state["scout_messages"]
    ))

    logger.info(
        "llm_call produced response tool_calls=%s num_tool_calls=%d",
        bool(response.tool_calls),
        len(response.tool_calls or []),
    )
    return {
        "scout_messages": [response]
    }

async def tool_node(state: ScoutState):
    """根据前一次大模型结果执行所有工具调用"""

    tool_calls = state["scout_messages"][-1].tool_calls
    used = state.get("tool_call_iterations", 0)
    remaining = max(0, MAX_TOOL_CALL_ITERATIONS - used)
    logger.info("tool_node executing %d tool calls", len(tool_calls or []))

    runtime_tools = await get_scout_tools()
    tools_by_name = {tool.name: tool for tool in runtime_tools}

    async def invoke_tool(tool_call: dict):
        tool_name = tool_call["name"]
        tool = tools_by_name.get(tool_name)
        if tool is None:
            return f"Tool error: unknown tool '{tool_name}'"

        logger.info("Invoking tool %s with args=%s", tool_name, tool_call["args"])
        try:
            # MCP StructuredTool 原生支持 ainvoke；同步工具由 LangChain 在线程中执行。
            return await tool.ainvoke(tool_call["args"])
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            return f"Tool error from {tool_name}: {exc}"

    # 同一轮彼此独立的 Tavily/MCP 工具调用并行执行。
    selected = tool_calls[:remaining]
    observations = list(await asyncio.gather(*(invoke_tool(tool_call) for tool_call in selected)))
    observations.extend("Tool call skipped: Scout tool budget exhausted." for _ in tool_calls[remaining:])

    def normalize_content(value) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    # 获取工具输出
    tool_outputs = [
        ToolMessage(
            content=normalize_content(observation),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {
        "scout_messages": tool_outputs,
        "tool_call_iterations": used + len(selected),
    }

def _invoke_compress_with_retry(messages, attempts: int = 5, base_delay: float = 8.0):
    """同步压缩调用的指数退避重试，应对 429 限流。"""
    import time

    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return compress_model.invoke(messages)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


def compress_intel(state: ScoutState) -> dict:
    """把搜集到的情报压缩为高价值摘要，只保留有用信息（价格/时间/班次/地址等）."""

    # 组装 prompt
    system_message = COMPRESS_INTEL_SYSTEM_PROMPT.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("scout_messages", []) +\
            [HumanMessage(content=COMPRESS_INTEL_HUMAN_PROMPT.format(travel_topic=state.get("travel_topic", "")))]
    logger.info("compress_intel invoked with %d messages", len(messages))

    # 调用 summary 模型（同步节点内直接调用，LLM client 已带 SDK 级重试）
    response = _invoke_compress_with_retry(messages)

    # 从 messages 和 tools 抽取 raw intel
    raw_intel = [
        str(m.content) for m in filter_messages(
            state["scout_messages"],
            include_types=["tool", "ai"]
        )
    ]

    logger.debug("compress_intel produced raw_intel_count=%d", len(raw_intel))
    return {
        "compressed_intel": str(response.content),
        "raw_intel": ["\n".join(raw_intel)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ScoutState) -> Literal["tool_node", "compress_intel"]:
    """决定是继续搜集情报还是输出最终情报。"""
    messages = state["scout_messages"]
    last_message = messages[-1]

    decision = "tool_node" if last_message.tool_calls else "compress_intel"
    logger.info("should_continue decision=%s (has_tool_calls=%s)", decision, bool(last_message.tool_calls))
    return decision


def after_tools(state: ScoutState) -> Literal["llm_call", "compress_intel"]:
    return "compress_intel" if state.get("tool_call_iterations", 0) >= MAX_TOOL_CALL_ITERATIONS else "llm_call"


# ===== GRAPH CONSTRUCTION =====

# Build the agent
scout_builder = StateGraph(ScoutState, output_schema=ScoutOutputState)

# Add nodes to the graph
scout_builder.add_node("llm_call", llm_call)
scout_builder.add_node("tool_node", tool_node)
scout_builder.add_node("compress_intel", compress_intel)

# Add edges to connect nodes
scout_builder.add_edge(START, "llm_call")
scout_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue intel gathering loop
        "compress_intel": "compress_intel", # 返回 final intel
    },
)
scout_builder.add_conditional_edges(
    "tool_node",
    after_tools,
    {"llm_call": "llm_call", "compress_intel": "compress_intel"},
)
scout_builder.add_edge("compress_intel", END)

# Compile the agent
scout_agent = scout_builder.compile()

if __name__ == "__main__":
    print(scout_agent.get_graph().draw_ascii())

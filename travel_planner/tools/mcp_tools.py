#***********************************************
#      Filename: mcp_tools.py
#   Description: 将旅游 MCP 服务动态转换为 LangChain 工具
#***********************************************

"""为 Scout Agent 加载高德地图 MCP 工具。

MCP 服务仍以 stdio 子进程运行；langchain-mcp-adapters 负责读取工具 schema，
并把服务端工具转换成可直接绑定到聊天模型的 LangChain StructuredTool。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from travel_planner import logging as tp_logging
from travel_planner.utils import load_config


logger = tp_logging.get_logger(__name__)

DEFAULT_STAGE = "prod"
DEFAULT_SERVER_MODULE = "travel_planner.mcp.travel_server"
MCP_SERVER_NAME = "travel_maps"

_MCP_TOOLS_CACHE: list[BaseTool] | None = None


def _load_mcp_config() -> dict:
    """读取当前 stage 的 MCP 配置；缺少配置时使用本地服务默认值。"""

    stage = os.environ.get("STAGE") or DEFAULT_STAGE
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    try:
        stage_config = load_config(stage_name=stage, config_path=config_path) or {}
    except (FileNotFoundError, KeyError, TypeError):
        logger.warning(
            "Unable to load MCP config from '%s' (stage=%s); using defaults",
            config_path,
            stage,
        )
        return {}
    return stage_config.get("mcp") or {}


def build_travel_mcp_connection() -> dict:
    """构造 langchain-mcp-adapters 使用的 stdio 连接配置。"""

    mcp_config = _load_mcp_config()
    server_module = mcp_config.get("server_module") or DEFAULT_SERVER_MODULE

    # stdio 子进程继承系统环境；显式 stage 配置覆盖父进程中的陈旧值。
    child_env = os.environ.copy()
    configured_api_key = mcp_config.get("amap_api_key")
    if configured_api_key:
        child_env["AMAP_MAPS_API_KEY"] = str(configured_api_key)

    project_root = Path(__file__).resolve().parents[2]
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", str(server_module)],
        "cwd": str(project_root),
        "env": child_env,
        "encoding": "utf-8",
        "encoding_error_handler": "replace",
    }


async def get_travel_mcp_tools() -> list[BaseTool]:
    """懒加载并缓存高德地图 MCP 工具。

    首次调用会启动一次 MCP 会话读取服务端工具 schema；转换后的工具在实际调用时
    由适配器自动创建短生命周期会话，因此不需要在 LangGraph 外维护连接。
    """

    global _MCP_TOOLS_CACHE
    if _MCP_TOOLS_CACHE is not None:
        return _MCP_TOOLS_CACHE

    client = MultiServerMCPClient(
        {MCP_SERVER_NAME: build_travel_mcp_connection()},
        handle_tool_errors=True,
    )
    tools = await client.get_tools(server_name=MCP_SERVER_NAME)
    _MCP_TOOLS_CACHE = tools
    logger.info(
        "Loaded %d travel MCP tools: %s",
        len(tools),
        ", ".join(tool.name for tool in tools),
    )
    return tools


def clear_travel_mcp_tools_cache() -> None:
    """清除 MCP 工具缓存，供配置热更新和测试使用。"""

    global _MCP_TOOLS_CACHE
    _MCP_TOOLS_CACHE = None

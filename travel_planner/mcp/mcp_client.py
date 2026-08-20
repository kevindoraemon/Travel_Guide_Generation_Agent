#***********************************************
#      Filename: mcp_client.py
#   Description: 旅游路书 MCP 客户端封装
#***********************************************
#
# 封装 MCP 客户端，连接 travel_server.py，提供同步/异步工具调用接口。
# 可独立运行测试，也可被多智能体工作流调用。
#

import asyncio
import subprocess
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from travel_planner.tools.mcp_tools import build_travel_mcp_connection


# 默认服务端脚本路径（travel_planner.mcp.travel_server）
DEFAULT_SERVER_MODULE = "travel_planner.mcp.travel_server"


class TravelMCPClient:
    """旅游路书 MCP 客户端，封装连接/调用/资源清理。

    用法：
        async with TravelMCPClient() as client:
            await client.connect()
            result = await client.call_tool("maps_weather", {"city": "北京"})
    """

    def __init__(self, server_module: str = DEFAULT_SERVER_MODULE):
        """初始化 MCP 客户端。

        Args:
            server_module: MCP 服务端模块路径（默认 travel_planner.mcp.travel_server）
        """
        self.server_module = server_module
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._connected = False

    async def connect(self) -> List[str]:
        """连接到 MCP 服务端，返回可用工具名列表。

        Returns:
            工具名称列表
        """
        if self._connected:
            return []

        connection = build_travel_mcp_connection()
        server_params = StdioServerParameters(
            command=connection["command"],
            args=["-m", self.server_module],
            env=connection["env"],
            cwd=connection["cwd"],
            encoding=connection["encoding"],
            encoding_error_handler=connection["encoding_error_handler"],
        )

        # 启动 MCP 服务端并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params, errlog=subprocess.DEVNULL)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # 列出可用工具
        response = await self.session.list_tools()
        tool_names = [tool.name for tool in response.tools]
        self._connected = True
        print(f"[MCP] 已连接服务端，可用工具: {tool_names}")
        return tool_names

    async def call_tool(self, function_name: str, tool_args: Dict[str, Any]) -> str:
        """调用 MCP 工具。

        Args:
            function_name: 工具名（如 maps_weather）
            tool_args: 工具参数字典（如 {"city": "北京", "date": "2026-07-20"}）

        Returns:
            工具返回的文本结果
        """
        if not self.session:
            raise RuntimeError("尚未连接 MCP 服务端，请先调用 connect()")

        try:
            result = await self.session.call_tool(function_name, tool_args)
            print(f"\n[MCP] 调用工具 {function_name}，参数 {tool_args}")
            content_text = result.content[0].text if result.content else "无返回内容"
            return content_text
        except Exception as e:
            print(f"[MCP] 调用工具 {function_name} 失败: {str(e)}")
            return f'{{"error": "{str(e)}"}}'

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出服务端提供的所有工具及其 schema。"""
        if not self.session:
            raise RuntimeError("尚未连接 MCP 服务端，请先调用 connect()")

        response = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    async def close(self):
        """清理资源，关闭连接。"""
        await self.exit_stack.aclose()
        self._connected = False
        self.session = None

    # 支持 async with 上下文管理
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ===== 独立运行测试 =====

async def _demo():
    """快速测试：连接服务端并查询北京天气。"""
    client = TravelMCPClient()
    try:
        tools = await client.connect()
        print(f"\n共 {len(tools)} 个工具可用\n")

        # 测试天气查询
        print("===== 测试 maps_weather =====")
        weather = await client.call_tool(
            "maps_weather", {"city": "北京", "date": "2026-07-20"}
        )
        print(weather)

        # 测试 POI 搜索
        print("\n===== 测试 maps_text_search =====")
        pois = await client.call_tool(
            "maps_text_search",
            {"keywords": "故宫博物院", "city": "北京", "top_k": 3},
        )
        print(pois)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_demo())

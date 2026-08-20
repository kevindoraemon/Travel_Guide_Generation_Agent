"""MCP 与 Scout 工具循环的集成测试（不调用外部地图 API 或 LLM）。"""

import unittest

from langchain_core.messages import AIMessage

import travel_planner.agents.scout_agent as scout_module


class ScoutMCPIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_scout_loads_local_and_mcp_tools(self):
        tools = await scout_module.get_scout_tools()
        tool_names = {tool.name for tool in tools}

        self.assertIn("tavily_search", tool_names)
        self.assertIn("think_tool", tool_names)
        self.assertIn("maps_weather", tool_names)
        self.assertIn("maps_text_search", tool_names)
        self.assertIn("maps_distance", tool_names)

    async def test_tool_node_returns_unknown_tool_error_to_model(self):
        message = AIMessage(
            content="",
            tool_calls=[{
                "name": "missing_test_tool",
                "args": {},
                "id": "test-call",
                "type": "tool_call",
            }],
        )

        result = await scout_module.tool_node({"scout_messages": [message]})

        tool_message = result["scout_messages"][0]
        self.assertIn("unknown tool", tool_message.content)
        self.assertEqual("test-call", tool_message.tool_call_id)

    def test_scout_stops_at_hard_tool_budget(self):
        self.assertEqual("compress_intel", scout_module.after_tools({"tool_call_iterations": 5}))
        self.assertEqual("llm_call", scout_module.after_tools({"tool_call_iterations": 4}))


if __name__ == "__main__":
    unittest.main()

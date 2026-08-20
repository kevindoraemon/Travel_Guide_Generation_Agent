#***********************************************
#      Filename: __init__.py
#   Description: 旅游路书 MCP 服务（基于 Model Context Protocol）
#***********************************************
#
# travel_planner.mcp —— 为旅游路书生成系统提供 MCP 工具服务
#
# 组成：
#   - travel_server：旅游 MCP 服务端，基于高德地图 API，提供天气/POI/路径规划/距离测算等工具
#   - mcp_client：MCP 客户端封装，连接服务端并调用工具
#
# 传输方式：stdio（标准输入输出）

"""
工具适配器 - 实现工具端口
合并了 ToolManager 的逻辑
"""

from typing import Any, List
from langchain.tools import tool
from domain.shared.ports.tool_port import ToolPort
from infrastructure.external.tool.tool_shema import WeatherQuery
from infrastructure.external.tool.amap_weather_query import exec_get_weather


class ToolAdapter(ToolPort):
    """工具适配器实现"""

    def init_tools(self) -> List[Any]:
        """初始化工具列表"""
        @tool("get_weather", description="根据城市名称获取天气", args_schema=WeatherQuery)
        def get_weather(city_name: str) -> str:
            return exec_get_weather(city_name)
        return [get_weather]

    def exec_tool(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        if tool_name == "get_weather":
            return exec_get_weather(args["city_name"])
        else:
            return f"未知工具: {tool_name}"

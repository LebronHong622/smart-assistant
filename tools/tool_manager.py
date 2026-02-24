from langchain.tools import tool
from .tool_shema import WeatherQuery
from .amap_weather_query import exec_get_weather

class ToolManager:
    def init_tools(self) -> list:
        """初始化工具"""
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
    
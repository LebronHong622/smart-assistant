"""
工具适配器 - 实现工具端口
合并了 ToolManager 的逻辑，支持动态工具加载
"""
import yaml
from pathlib import Path
from typing import Any, List, Dict
from langchain.tools import tool, BaseTool
from domain.shared.ports.tool_port import ToolPort
from infrastructure.external.tool.tools.amap_weather_query.schema import WeatherQuery
from infrastructure.external.tool.tools.amap_weather_query.tool import exec_get_weather
from infrastructure.external.tool.tools.document_retrieval.langchain import langchain_document_retrieval
from infrastructure.external.tool.tools.document_retrieval.standard import document_retrieval
from infrastructure.core.log import app_logger


class ToolAdapter(ToolPort):
    """工具适配器实现"""

    def __init__(self):
        self.config_path = Path(__file__).parent.parent.parent.parent / "config" / "tools" / "agent_tools.yaml"
        self.tool_config = self._load_tool_config()
        self.tool_registry = self._build_tool_registry()

    def _load_tool_config(self) -> Dict[str, List[str]]:
        """加载工具配置"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            app_logger.info(f"工具配置加载成功，支持的Agent类型: {list(config.keys())}")
            return config
        except Exception as e:
            app_logger.error(f"工具配置加载失败: {str(e)}，使用默认配置")
            return {"default": ["get_weather", "document_retrieval"]}

    def _build_tool_registry(self) -> Dict[str, BaseTool]:
        """构建工具注册表"""
        @tool("get_weather", description="根据城市名称获取天气", args_schema=WeatherQuery)
        def get_weather(city_name: str) -> str:
            return exec_get_weather(city_name)

        return {
            "get_weather": get_weather,
            "document_retrieval": document_retrieval,
            "langchain_document_retrieval": langchain_document_retrieval
        }

    def get_tools(self, agent_type: str = "default") -> List[BaseTool]:
        """
        根据Agent类型获取对应的工具列表
        Args:
            agent_type: Agent类型，默认default
        Returns:
            工具列表
        """
        tool_names = self.tool_config.get(agent_type, self.tool_config.get("default", []))
        tools = [self.tool_registry[name] for name in tool_names if name in self.tool_registry]
        app_logger.info(f"为Agent类型 {agent_type} 加载了 {len(tools)} 个工具: {[tool.name for tool in tools]}")
        return tools

    def init_tools(self) -> List[Any]:
        """初始化工具列表（兼容旧接口）"""
        return self.get_tools("default")

    def exec_tool(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        if tool_name in self.tool_registry:
            return self.tool_registry[tool_name].invoke(args)
        else:
            app_logger.warning(f"尝试执行未知工具: {tool_name}")
            return f"未知工具: {tool_name}"

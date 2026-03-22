"""
LangChain框架适配器实现
专门为LangChain框架实现的工具端口，直接返回LangChain原生格式工具
"""
from typing import Dict, List, Any
from langchain.tools import tool, BaseTool
from infrastructure.external.tool.adapters.base_frame_adapter import BaseFrameAdapter
from infrastructure.external.tool.tools.amap_weather_query.schema import WeatherQuery
from infrastructure.external.tool.tools.amap_weather_query.tool import exec_get_weather
from infrastructure.external.tool.tools.document_retrieval.langchain import langchain_document_retrieval
from infrastructure.external.tool.tools.document_retrieval.standard import document_retrieval
from infrastructure.core.log import app_logger


class LangChainFrameAdapter(BaseFrameAdapter):
    """LangChain框架适配器实现"""

    def _build_tool_registry(self) -> Dict[str, BaseTool]:
        """构建LangChain工具注册表，所有工具直接符合LangChain BaseTool格式"""
        return {
            "langchain_document_retrieval": langchain_document_retrieval
        }

    def get_tools(self, agent_type: str = "default", return_raw: bool = False, **kwargs) -> List[BaseTool]:
        """
        获取LangChain原生格式的工具列表，可直接被LangChain框架使用

        Args:
            agent_type: Agent类型，默认default
            return_raw: 是否返回原始函数，默认False返回BaseTool实例列表
            **kwargs: 其他参数

        Returns:
            LangChain BaseTool格式的工具列表
        """
        # 获取该Agent类型的工具名称列表
        tool_names = self.tool_config.get(agent_type, self.tool_config.get("default", []))

        # 过滤并返回工具列表
        tools = []
        for name in tool_names:
            if name in self.tool_registry:
                tool = self.tool_registry[name]
                if return_raw:
                    tools.append(tool.func)
                else:
                    tools.append(tool)
            else:
                app_logger.warning(f"工具 {name} 未在注册表中找到，已跳过")

        app_logger.info(f"为Agent类型 {agent_type} 加载了 {len(tools)} 个LangChain工具: {[tool.name for tool in tools]}")
        return tools
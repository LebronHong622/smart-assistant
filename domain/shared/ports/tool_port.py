"""
工具端口 - 定义工具接口
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict


class ToolPort(ABC):
    """工具接口"""

    @abstractmethod
    def get_tools(self, agent_type: str = "default", framework: str = "langchain") -> List[Any]:
        """
        根据Agent类型和框架获取对应的工具列表

        Args:
            agent_type: Agent类型，默认default
            framework: 框架类型，默认langchain

        Returns:
            框架原生格式的工具列表
        """
        pass

    @abstractmethod
    def exec_tool(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def reload_config(self) -> None:
        """重新加载配置"""
        pass

"""
工具端口 - 定义工具接口
"""

from abc import ABC, abstractmethod
from typing import List, Any


class ToolPort(ABC):
    """工具接口"""

    @abstractmethod
    def init_tools(self) -> List[Any]:
        """初始化工具列表"""
        pass

    @abstractmethod
    def exec_tool(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_tools(self, agent_type: str = "default") -> List[Any]:
        """根据Agent类型获取对应的工具列表
        
        Args:
            agent_type: Agent类型，默认default
            
        Returns:
            工具列表
        """
        pass

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

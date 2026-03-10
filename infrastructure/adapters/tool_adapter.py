"""
工具适配器 - 实现工具端口
"""

from domain.shared.ports.tool_port import ToolPort
from infrastructure.external.tool.tool_manager import ToolManager


class ToolAdapter(ToolPort):
    """工具适配器实现"""

    def __init__(self):
        self._tool_manager = ToolManager()

    def init_tools(self) -> list[any]:
        """初始化工具列表"""
        return self._tool_manager.init_tools()

    def exec_tool(self, tool_name: str, args: dict) -> str:
        """执行工具"""
        return self._tool_manager.exec_tool(tool_name, args)

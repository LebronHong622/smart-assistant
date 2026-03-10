"""
工具工厂 - 管理工具适配器
"""

from typing import Dict, Type, Any, List
from domain.shared.ports.tool_port import ToolPort


class ToolFactory:
    """工具工厂"""

    _adapter_class: Type[ToolPort] = None
    _instance: ToolPort = None

    @classmethod
    def register(cls, adapter_class: Type[ToolPort]):
        """注册工具适配器"""
        cls._adapter_class = adapter_class

    @classmethod
    def get(cls) -> ToolPort:
        """获取工具适配器实例"""
        if cls._instance is None:
            if cls._adapter_class is None:
                raise ValueError("未注册工具适配器")
            cls._instance = cls._adapter_class()
        return cls._instance

    @classmethod
    def init_tools(cls) -> List[Any]:
        """初始化工具列表"""
        return cls.get().init_tools()

    @classmethod
    def exec_tool(cls, tool_name: str, args: dict) -> str:
        """执行工具"""
        return cls.get().exec_tool(tool_name, args)


# 默认注册工具适配器（延迟导入）
def _register_default():
    from infrastructure.external.tool.adapters.tool_adapter import ToolAdapter
    ToolFactory.register(ToolAdapter)

_register_default()
del _register_default

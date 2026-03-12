"""
基础框架适配器 - 抽象基类，继承ToolPort接口
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from domain.shared.ports.tool_port import ToolPort
from infrastructure.external.tool.loaders.config_loader_port import ToolConfigLoaderPort
from infrastructure.log import app_logger


class BaseFrameAdapter(ToolPort, ABC):
    """基础框架适配器抽象类，所有框架实现的基类"""

    def __init__(self, config_loader: ToolConfigLoaderPort):
        """
        构造函数，通过依赖注入方式传入配置加载器

        Args:
            config_loader: 配置加载器实例
        """
        self.config_loader = config_loader
        self.tool_config = self._load_tool_config()
        self.tool_registry = self._build_tool_registry()

    @abstractmethod
    def _build_tool_registry(self) -> Dict[str, Any]:
        """抽象方法：构建工具注册表，由具体框架子类实现"""
        pass

    @abstractmethod
    def get_tools(self, agent_type: str = "default", **kwargs) -> List[Any]:
        """
        抽象方法：获取框架原生格式的工具列表，由具体框架子类实现

        Args:
            agent_type: Agent类型，默认default
            **kwargs: 框架特定参数

        Returns:
            框架原生格式的工具列表
        """
        pass

    def _load_tool_config(self) -> Dict[str, List[str]]:
        """加载工具配置（通用实现）"""
        return self.config_loader.load_config()

    def exec_tool(self, tool_name: str, args: dict) -> str:
        """执行工具（通用实现）"""
        if tool_name in self.tool_registry:
            return self.tool_registry[tool_name].invoke(args)
        else:
            app_logger.warning(f"尝试执行未知工具: {tool_name}")
            return f"未知工具: {tool_name}"

    def reload_config(self) -> None:
        """重新加载配置（通用实现）"""
        self.tool_config = self._load_tool_config()
        app_logger.info("工具配置已重新加载")
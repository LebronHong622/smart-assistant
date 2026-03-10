"""
组件注册表，统一管理所有组件
使用注册机制，支持动态添加和获取组件
"""
from typing import Dict, Optional, Type
from infrastructure.core.log import app_logger


class ComponentRegistry:
    """
    组件注册表（单例模式）
    负责管理所有组件的注册和获取
    """
    _instance: Optional["ComponentRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "ComponentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._components: Dict[str, Type] = {}
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ComponentRegistry":
        """获取单例实例"""
        return cls()

    def register(self, component_class: Type, component_name: str) -> None:
        """
        注册组件类
        :param component_class: 组件类
        :param component_name: 组件名称
        """
        if component_name in self._components:
            app_logger.warning(f"组件 {component_name} 已注册，将被覆盖")
        self._components[component_name] = component_class
        app_logger.info(f"组件 {component_name} 注册成功")

    def unregister(self, component_name: str) -> None:
        """
        注销组件
        :param component_name: 组件名称
        """
        if component_name in self._components:
            del self._components[component_name]
            app_logger.info(f"组件 {component_name} 注销成功")
        else:
            app_logger.warning(f"组件 {component_name} 未注册，无需注销")

    def get_component_class(self, component_name: str) -> Optional[Type]:
        """
        获取组件类
        :param component_name: 组件名称
        :return: 组件类，如果不存在返回None
        """
        return self._components.get(component_name)

    def has_component(self, component_name: str) -> bool:
        """
        检查组件是否已注册
        :param component_name: 组件名称
        :return: 是否已注册
        """
        return component_name in self._components

    def list_components(self) -> list[str]:
        """
        列出所有已注册的组件名称
        :return: 组件名称列表
        """
        return list(self._components.keys())

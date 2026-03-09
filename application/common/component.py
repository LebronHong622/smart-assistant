"""
组件抽象基类，定义所有组件的统一接口
遵循适配器模式，将具体的客户端适配为统一接口
"""
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum


class ComponentStatus(Enum):
    """组件状态枚举"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class Component(ABC):
    """
    组件抽象基类
    定义所有组件必须实现的标准接口
    """

    def __init__(self, component_name: str):
        """
        初始化组件
        :param component_name: 组件名称
        """
        self.name = component_name
        self._status = ComponentStatus.NOT_INITIALIZED
        self._instance: Optional[object] = None

    @abstractmethod
    def initialize(self) -> None:
        """
        初始化组件
        具体实现由子类提供
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        关闭组件，释放资源
        具体实现由子类提供
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        健康检查
        :return: 组件是否健康
        """
        pass

    @abstractmethod
    def get_instance(self) -> object:
        """
        获取组件实例
        :return: 组件的具体实例
        """
        pass

    def get_status(self) -> ComponentStatus:
        """
        获取组件状态
        :return: 组件状态
        """
        return self._status

    def get_name(self) -> str:
        """
        获取组件名称
        :return: 组件名称
        """
        return self.name

    def _set_status(self, status: ComponentStatus) -> None:
        """
        设置组件状态
        :param status: 组件状态
        """
        self._status = status

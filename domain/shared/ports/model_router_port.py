from abc import ABC, abstractmethod
from typing import Any
from domain.shared.model_enums import ModelType, RoutingStrategy
from domain.shared.ports.model_capability_port import BaseModel


class BaseRoutingStrategy(ABC):
    """路由策略抽象基类，定义不同框架下的模型创建逻辑"""

    @abstractmethod
    def select_model(
        self,
        model_type: ModelType,
        **kwargs
    ) -> BaseModel[Any, Any]:
        """根据模型类型创建对应模型实例"""
        pass


class ModelRouterPort(ABC):
    """模型路由接口，根据模型类型和策略返回对应模型"""

    @abstractmethod
    def get_model(
        self,
        model_type: ModelType,
        strategy: RoutingStrategy = RoutingStrategy.DEFAULT,
        framework: str = "langchain",
        **kwargs
    ) -> BaseModel[Any, Any]:
        """根据模型类型、路由策略和框架获取模型实例"""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """清空模型缓存"""
        pass

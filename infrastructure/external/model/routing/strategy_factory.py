from typing import Dict, Type, Tuple
from domain.shared.model_enums import ModelType, RoutingStrategy
from domain.shared.ports.model_router_port import BaseRoutingStrategy
from infrastructure.external.model.routing.strategies.langchain_strategy import (
    LangChainChatStrategy,
    LangChainToolChatStrategy,
    LangChainEmbeddingStrategy
)
from infrastructure.core.log import app_logger


class StrategyFactory:
    """策略工厂类，负责根据框架、模型类型和路由策略创建对应策略实例"""

    # 策略注册表: (framework, model_type, strategy) -> 策略类
    _strategy_registry: Dict[Tuple[str, ModelType, RoutingStrategy], Type[BaseRoutingStrategy]] = {}

    @classmethod
    def register_strategy(
        cls,
        framework: str,
        model_type: ModelType,
        strategy: RoutingStrategy,
        strategy_class: Type[BaseRoutingStrategy]
    ) -> None:
        """注册新的策略实现"""
        key = (framework.lower(), model_type, strategy)
        cls._strategy_registry[key] = strategy_class
        app_logger.debug(f"注册策略: framework={framework}, model_type={model_type.value}, strategy={strategy.value}")

    @classmethod
    def get_strategy(
        cls,
        framework: str,
        model_type: ModelType,
        strategy: RoutingStrategy
    ) -> BaseRoutingStrategy:
        """根据参数获取对应策略实例"""
        key = (framework.lower(), model_type, strategy)

        if key not in cls._strategy_registry:
            raise ValueError(
                f"未找到对应策略: framework={framework}, model_type={model_type.value}, strategy={strategy.value}"
            )

        strategy_class = cls._strategy_registry[key]
        return strategy_class()


# 注册默认的LangChain策略
StrategyFactory.register_strategy(
    "langchain",
    ModelType.CHAT,
    RoutingStrategy.DEFAULT,
    LangChainChatStrategy
)

StrategyFactory.register_strategy(
    "langchain",
    ModelType.EMBEDDING,
    RoutingStrategy.DEFAULT,
    LangChainEmbeddingStrategy
)
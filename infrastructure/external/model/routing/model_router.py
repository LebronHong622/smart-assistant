from typing import Any, Dict
from domain.shared.model_enums import ModelType, RoutingStrategy
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.ports.model_capability_port import BaseModel
from infrastructure.external.model.routing.strategy_factory import StrategyFactory
from infrastructure.core.log import app_logger
from config.settings import settings


class ModelRouter(ModelRouterPort):
    """模型路由实现类，单例模式"""

    _instance = None
    _initialized = False
    _model_cache: Dict[str, BaseModel[Any, Any]] = {}
    _default_framework: str = "langchain"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # 从配置获取默认框架
            self._default_framework = getattr(settings, "DEFAULT_MODEL_FRAMEWORK", "langchain")
            app_logger.info(f"初始化模型路由，默认框架: {self._default_framework}")

    def get_model(
        self,
        model_type: ModelType,
        strategy: RoutingStrategy = RoutingStrategy.DEFAULT,
        framework: str = None,
        **kwargs
    ) -> BaseModel[Any, Any]:
        """根据模型类型、路由策略和框架获取模型实例"""
        # 使用指定框架或默认框架
        target_framework = framework or self._default_framework

        # 生成缓存key，包含框架信息
        cache_key = f"{target_framework}_{model_type.value}_{strategy.value}"
        # 如果有额外的模型名称参数，也加入缓存key
        if "model_name" in kwargs:
            cache_key += f"_{kwargs['model_name']}"

        # 优先从缓存获取
        if cache_key in self._model_cache:
            app_logger.debug(f"从缓存获取模型: {cache_key}")
            return self._model_cache[cache_key]

        # 获取对应策略
        strategy_impl = StrategyFactory.get_strategy(target_framework, model_type, strategy)

        # 使用策略创建模型
        model: BaseModel[Any, Any] = strategy_impl.select_model(model_type, **kwargs)

        # 存入缓存
        self._model_cache[cache_key] = model
        app_logger.info(
            f"创建模型实例: framework={target_framework}, type={model_type.value}, "
            f"strategy={strategy.value}, cache_key={cache_key}"
        )

        return model

    def clear_cache(self) -> None:
        """清空模型缓存"""
        self._model_cache.clear()
        app_logger.info("模型缓存已清空")

"""
LLM 工厂 - 管理LLM模型适配器
"""

from typing import Dict, Type, Any, Optional
from domain.shared.ports.model_port import ModelPort


class LLMFactory:
    """LLM 工厂"""

    _adapters: Dict[str, Type[ModelPort]] = {}
    _instances: Dict[str, ModelPort] = {}

    @classmethod
    def register(cls, name: str, adapter_class: Type[ModelPort]):
        """注册LLM适配器"""
        cls._adapters[name] = adapter_class

    @classmethod
    def get(cls, name: str = "default") -> ModelPort:
        """获取LLM适配器实例"""
        if name not in cls._instances:
            if name not in cls._adapters:
                raise ValueError(f"未注册的LLM适配器: {name}")
            cls._instances[name] = cls._adapters[name]()
        return cls._instances[name]

    @classmethod
    def get_model(cls, adapter_name: str = "default", model_name: Optional[str] = None) -> Any:
        """获取具体模型实例"""
        adapter = cls.get(adapter_name)
        if model_name:
            return adapter.get_model(model_name)
        return adapter.get_default_model()

    @classmethod
    def list_adapters(cls) -> list[str]:
        """列出所有已注册的适配器"""
        return list(cls._adapters.keys())


# 默认注册 LLM 适配器
def _register_default():
    from infrastructure.external.model.llm.adapters.llm_adapter import LLMAdapter
    LLMFactory.register("default", LLMAdapter)

_register_default()
del _register_default

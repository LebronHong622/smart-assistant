"""
嵌入向量工厂 - 管理嵌入向量适配器
"""

from typing import Dict, Type
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort


class EmbeddingFactory:
    """嵌入向量工厂"""

    _adapters: Dict[str, Type[EmbeddingGeneratorPort]] = {}
    _instances: Dict[str, EmbeddingGeneratorPort] = {}

    @classmethod
    def register(cls, name: str, adapter_class: Type[EmbeddingGeneratorPort]):
        """注册嵌入向量适配器"""
        cls._adapters[name] = adapter_class

    @classmethod
    def get(cls, name: str = "dashscope") -> EmbeddingGeneratorPort:
        """获取嵌入向量适配器实例"""
        if name not in cls._instances:
            if name not in cls._adapters:
                raise ValueError(f"未注册的嵌入向量适配器: {name}")
            cls._instances[name] = cls._adapters[name]()
        return cls._instances[name]

    @classmethod
    def list_adapters(cls) -> list[str]:
        """列出所有已注册的适配器"""
        return list(cls._adapters.keys())


# 默认注册 DashScope 适配器
def _register_default():
    from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter
    EmbeddingFactory.register("dashscope", DashScopeEmbeddingAdapter)

_register_default()
del _register_default

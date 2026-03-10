"""
向量存储工厂 - 管理向量存储适配器
"""

from typing import Dict, Type, Optional
from domain.shared.ports.vector_store_port import VectorStorePort


class VectorStoreFactory:
    """向量存储工厂"""

    _adapters: Dict[str, Type[VectorStorePort]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: Type[VectorStorePort]):
        """注册向量存储适配器"""
        cls._adapters[name] = adapter_class

    @classmethod
    def create(cls, name: str = "milvus", **kwargs) -> VectorStorePort:
        """创建向量存储适配器实例"""
        if name not in cls._adapters:
            raise ValueError(f"未注册的向量存储适配器: {name}")
        return cls._adapters[name](**kwargs)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """列出所有已注册的适配器"""
        return list(cls._adapters.keys())


# 默认注册 Milvus 适配器
from infrastructure.persistence.vector.adapters.milvus_adapter import MilvusAdapter
VectorStoreFactory.register("milvus", MilvusAdapter)

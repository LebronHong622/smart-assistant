"""
向量存储工厂
根据配置动态创建向量存储实例

已重构为 RAGComponentFactory 的代理，保持向后兼容
"""

from typing import Any, List, Optional, Union

from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory


class VectorStoreFactory:
    """
    向量存储工厂类
    
    根据配置动态创建不同类型的向量存储
    支持 Milvus、Chroma、FAISS、Qdrant 等
    
    注意: 此类现在是 RAGComponentFactory 的代理
    实际实现根据 settings.app.framework 配置动态决定
    """

    @classmethod
    def register_store(cls, name: str, store_class: Any) -> None:
        """注册自定义向量存储"""
        factory = RAGComponentFactory.get_vector_store_factory()
        if hasattr(factory, 'register_store'):
            factory.register_store(name, store_class)

    @classmethod
    def create_milvus_store(
        cls,
        embedding: Any,
        collection_name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """创建 Milvus 向量存储"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.create_milvus_store(embedding, collection_name, config, **kwargs)

    @classmethod
    def create_chroma_store(
        cls,
        embedding: Any,
        collection_name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """创建 Chroma 向量存储"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.create_chroma_store(embedding, collection_name, config, **kwargs)

    @classmethod
    def create_faiss_store(
        cls,
        embedding: Any,
        index_name: str = "faiss_index",
        config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """创建 FAISS 向量存储"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.create_faiss_store(embedding, index_name, config, **kwargs)

    @classmethod
    def create_qdrant_store(
        cls,
        embedding: Any,
        collection_name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """创建 Qdrant 向量存储"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.create_qdrant_store(embedding, collection_name, config, **kwargs)

    @classmethod
    def create_store(
        cls,
        embedding: Any,
        collection_name: str,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """创建向量存储实例"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.create_store(embedding, collection_name, provider, **kwargs)

    @classmethod
    def get_store_config(cls, provider: Optional[str] = None) -> Any:
        """获取向量存储配置"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.get_store_config(provider)

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """列出所有支持的向量存储类型"""
        factory = RAGComponentFactory.get_vector_store_factory()
        return factory.list_supported_providers()

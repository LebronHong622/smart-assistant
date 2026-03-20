"""
LangChain 向量存储工厂
根据 provider 分发调用不同的向量存储实现
"""
from typing import Any, Dict, List, Optional, Type, Callable
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from config.rag_settings import rag_settings
from domain.shared.ports import VectorStoreFactoryPort
from infrastructure.core.log import app_logger


class LangChainVectorStoreFactory(VectorStoreFactoryPort):
    """
    LangChain 向量存储工厂
    """

    _registered_stores: Dict[str, Type[VectorStore]] = {}

    @classmethod
    def register_store(cls, name: str, store_class: Type[VectorStore]) -> None:
        """注册自定义向量存储"""
        cls._registered_stores[name] = store_class
        app_logger.info(f"注册向量存储: {name}")

    @classmethod
    def create_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        provider: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        """创建向量存储实例"""
        vector_config = rag_settings.get_vector_config()
        provider = provider or vector_config.provider
        app_logger.info(f"创建向量存储: provider={provider}, collection={collection_name}")

        if provider in cls._registered_stores:
            store_class = cls._registered_stores[provider]
            return store_class(
                embedding_function=embedding,
                collection_name=collection_name,
                **kwargs,
            )

        # 延迟导入各个 provider 的创建函数
        creators: Dict[str, Callable] = {}
        if provider == "milvus":
            from infrastructure.rag.embeddings.langchain.vector_stores.milvus import create_milvus_store
            creators["milvus"] = create_milvus_store
        elif provider == "faiss":
            from infrastructure.rag.embeddings.langchain.vector_stores.faiss import create_faiss_store
            creators["faiss"] = create_faiss_store

        if provider not in creators:
            raise ValueError(f"不支持的向量存储类型: {provider}")

        return creators[provider](
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def get_store_config(cls, provider: Optional[str] = None) -> Any:
        """获取向量存储配置"""
        vector_config = rag_settings.get_vector_config()
        provider = provider or vector_config.provider
        return vector_config.get_active_config()

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """列出所有支持的向量存储类型"""
        providers = {"milvus", "faiss"}
        providers.update(cls._registered_stores.keys())
        return list(providers)

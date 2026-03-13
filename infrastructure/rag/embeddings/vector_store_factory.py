"""
向量存储工厂
根据配置动态创建向量存储实例
"""

from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config.rag_settings import (
    rag_settings,
    VectorConfig,
    MilvusConfig,
    ChromaConfig,
    FAISSConfig,
    QdrantConfig,
)
from infrastructure.core.log import app_logger


class VectorStoreFactory:
    """
    向量存储工厂类
    
    根据配置动态创建不同类型的向量存储
    支持 Milvus、Chroma、FAISS、Qdrant 等
    """

    # 已注册的向量存储类
    _registered_stores: Dict[str, Type[VectorStore]] = {}

    @classmethod
    def register_store(cls, name: str, store_class: Type[VectorStore]) -> None:
        """
        注册自定义向量存储
        
        Args:
            name: 向量存储名称
            store_class: 向量存储类
        """
        cls._registered_stores[name] = store_class
        app_logger.info(f"注册向量存储: {name}")

    @classmethod
    def create_milvus_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        config: Optional[MilvusConfig] = None,
        **kwargs,
    ) -> VectorStore:
        """
       创建 Milvus 向量存储
        
        Args:
            embedding: 嵌入函数
            collection_name: 集合名称
            config: Milvus 配置
            **kwargs: 额外参数
            
        Returns:
            Milvus 向量存储实例
        """
        from langchain_milvus import Milvus
        
        config = config or rag_settings.vector.milvus
        connection_args = {"uri": config.get_connection_uri()}
        
        # 合并 langchain 配置
        langchain_config = config.langchain_config
        
        return Milvus(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            auto_id=langchain_config.auto_id,
            vector_field=langchain_config.vector_field,
            text_field=langchain_config.text_field,
            metadata_field=langchain_config.metadata_field,
            **kwargs,
        )

    @classmethod
    def create_chroma_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        config: Optional[ChromaConfig] = None,
        **kwargs,
    ) -> VectorStore:
        """
        创建 Chroma 向量存储
        
        Args:
            embedding: 嵌入函数
            collection_name: 集合名称
            config: Chroma 配置
            **kwargs: 额外参数
            
        Returns:
            Chroma 向量存储实例
        """
        from langchain_chroma import Chroma
        
        config = config or rag_settings.vector.chroma
        
        return Chroma(
            embedding_function=embedding,
            collection_name=collection_name,
            persist_directory=config.persist_directory,
            client_settings=config.settings if config.settings else None,
            **kwargs,
        )

    @classmethod
    def create_faiss_store(
        cls,
        embedding: Embeddings,
        index_name: str = "faiss_index",
        config: Optional[FAISSConfig] = None,
        **kwargs,
    ) -> VectorStore:
        """
        创建 FAISS 向量存储
        
        Args:
            embedding: 嵌入函数
            index_name: 索引名称
            config: FAISS 配置
            **kwargs: 额外参数
            
        Returns:
            FAISS 向量存储实例
        """
        from langchain_community.vectorstores import FAISS
        
        config = config or rag_settings.vector.faiss
        
        return FAISS(
            embedding_function=embedding,
            index_name=index_name,
            **kwargs,
        )

    @classmethod
    def create_qdrant_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        config: Optional[QdrantConfig] = None,
        **kwargs,
    ) -> VectorStore:
        """
        创建 Qdrant 向量存储
        
        Args:
            embedding: 嵌入函数
            collection_name: 集合名称
            config: Qdrant 配置
            **kwargs: 额外参数
            
        Returns:
            Qdrant 向量存储实例
        """
        from langchain_qdrant import Qdrant
        
        config = config or rag_settings.vector.qdrant
        
        return Qdrant(
            embeddings=embedding,
            collection_name=collection_name,
            url=config.get_url(),
            api_key=config.connection.api_key or None,
            **kwargs,
        )

    @classmethod
    def create_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        provider: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        """
        创建向量存储实例
        
        Args:
            embedding: 嵌入函数
            collection_name: 集合名称
            provider: 向量存储提供者，默认从配置读取
            **kwargs: 额外参数
            
        Returns:
            向量存储实例
            
        Raises:
            ValueError: 不支持的向量存储类型
        """
        vector_config = rag_settings.get_vector_config()
        provider = provider or vector_config.provider
        app_logger.info(f"创建向量存储: provider={provider}, collection={collection_name}")

        # 检查是否已注册
        if provider in cls._registered_stores:
            store_class = cls._registered_stores[provider]
            return store_class(
                embedding_function=embedding,
                collection_name=collection_name,
                **kwargs,
            )

        # 内置支持
        creators = {
            "milvus": cls.create_milvus_store,
            "chroma": cls.create_chroma_store,
            "faiss": cls.create_faiss_store,
            "qdrant": cls.create_qdrant_store,
        }

        if provider not in creators:
            raise ValueError(f"不支持的向量存储类型: {provider}")

        return creators[provider](
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def get_store_config(cls, provider: Optional[str] = None) -> Union[MilvusConfig, ChromaConfig, FAISSConfig, QdrantConfig]:
        """
        获取向量存储配置
        
        Args:
            provider: 向量存储提供者
            
        Returns:
            向量存储配置
        """
        vector_config = rag_settings.get_vector_config()
        provider = provider or vector_config.provider
        return vector_config.get_active_config()

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """
        列出所有支持的向量存储类型
        
        Returns:
            向量存储类型列表
        """
        providers = {"milvus", "chroma", "faiss", "qdrant"}
        providers.update(cls._registered_stores.keys())
        return list(providers)

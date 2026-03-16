"""
向量存储仓库接口和 Milvus 具体实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from infrastructure.persistence.vector.milvus_client import milvus_client
from config.settings import settings
from infrastructure.persistence.vector.milvus_collections.collection_manager import CollectionSchemaConfig


class VectorStore(ABC):
    """向量存储仓库接口"""

    @abstractmethod
    def insert_documents(self, documents: List[Dict[str, Any]], **kwargs) -> None:
        """插入文档嵌入向量"""
        pass

    @abstractmethod
    def search_documents(self, query_embedding: List[float], limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """删除集合"""
        pass


class MilvusVectorStore(VectorStore):
    """Milvus 向量存储仓库实现"""

    def __init__(self, collection_name: str = None, schema_config: Optional[CollectionSchemaConfig] = None):
        self.collection_name = collection_name or settings.milvus.milvus_collection_name

        # 如果提供了 Schema 配置，注册到 MilvusClient
        if schema_config:
            if schema_config.collection_name != self.collection_name:
                raise ValueError(f"Schema 配置的 collection_name ({schema_config.collection_name}) 与传入的 collection_name ({self.collection_name}) 不一致")
            milvus_client.register_schema(schema_config)

    def insert_documents(self, documents: List[Dict[str, Any]], **kwargs) -> None:
        """插入文档嵌入向量"""
        milvus_client.insert_embeddings(documents, self.collection_name, **kwargs)

    def search_documents(self, query_embedding: List[float], limit: int = 5, anns_field: str = "embedding", **kwargs) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        return milvus_client.search_embeddings(query_embedding, limit, self.collection_name, anns_field, **kwargs)

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        return milvus_client.get_collection_info(self.collection_name)

    def delete_collection(self) -> None:
        """删除集合"""
        milvus_client.delete_collection(self.collection_name)

    def ensure_collection_exists(self) -> None:
        """确保集合存在"""
        milvus_client.ensure_collection_exists(self.collection_name)

    def get_collection_fields(self) -> list[str]:
        """获取集合字段列表"""
        from pymilvus import Collection
        collection = Collection(self.collection_name)
        return [field.name for field in collection.schema.fields]

    @staticmethod
    def list_all_collections() -> list[str]:
        """列出所有 Collection"""
        return milvus_client.list_collections()
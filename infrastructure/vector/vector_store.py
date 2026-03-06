"""
向量存储仓库接口和 Milvus 具体实现
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from infrastructure.vector.milvus_client import milvus_client
from infrastructure.config.settings import settings


class VectorStore(ABC):
    """向量存储仓库接口"""

    @abstractmethod
    def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """插入文档嵌入向量"""
        pass

    @abstractmethod
    def search_documents(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
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

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.milvus.milvus_collection_name

    def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """插入文档嵌入向量"""
        milvus_client.insert_embeddings(documents, self.collection_name)

    def search_documents(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        return milvus_client.search_embeddings(query_embedding, limit, self.collection_name)

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        return milvus_client.get_collection_info(self.collection_name)

    def delete_collection(self) -> None:
        """删除集合"""
        milvus_client.delete_collection(self.collection_name)
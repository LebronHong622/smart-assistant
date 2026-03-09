"""
向量存储适配器 - 实现向量存储端口
"""

from domain.port.vector_store_port import VectorStorePort
from infrastructure.vector.vector_store import MilvusVectorStore


class VectorStoreAdapter(VectorStorePort):
    """向量存储适配器实现"""

    def __init__(self, collection_name: str = None):
        self._vector_store = MilvusVectorStore(collection_name=collection_name)

    def insert_documents(self, documents: list[dict[str, any]]) -> None:
        """插入文档嵌入向量"""
        self._vector_store.insert_documents(documents)

    def search_documents(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, any]]:
        """搜索相似文档"""
        return self._vector_store.search_documents(query_embedding, limit)

    def get_collection_info(self) -> dict[str, any]:
        """获取集合信息"""
        return self._vector_store.get_collection_info()

    def get_collection_fields(self) -> list[str]:
        """获取集合字段列表"""
        return self._vector_store.get_collection_fields()

    def ensure_collection_exists(self) -> None:
        """确保集合存在"""
        self._vector_store.ensure_collection_exists()

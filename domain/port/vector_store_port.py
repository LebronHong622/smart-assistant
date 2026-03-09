"""
向量存储端口 - 定义向量存储接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorStorePort(ABC):
    """向量存储接口"""

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
    def get_collection_fields(self) -> List[str]:
        """获取集合字段列表"""
        pass

    @abstractmethod
    def ensure_collection_exists(self) -> None:
        """确保集合存在"""
        pass

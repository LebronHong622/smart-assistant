"""
文档检索服务接口
"""

from abc import ABC, abstractmethod
from typing import List
from domain.document.value_object.retrieval_result import RetrievalResult


class DocumentRetrievalService(ABC):
    """文档检索服务接口"""

    @abstractmethod
    def retrieve_similar_documents(self, query: str, limit: int = 5, score_threshold: float = 0.5, collection_name: str = None) -> List[RetrievalResult]:
        """
        根据查询文本检索相似文档

        Args:
            query: 查询文本
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值
            collection_name: 可选，指定检索的 Collection 名称

        Returns:
            检索结果列表
        """
        pass

    @abstractmethod
    def retrieve_similar_documents_by_embedding(self, query_embedding: List[float], limit: int = 5, score_threshold: float = 0.5, collection_name: str = None) -> List[RetrievalResult]:
        """
        根据查询向量检索相似文档

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值
            collection_name: 可选，指定检索的 Collection 名称

        Returns:
            检索结果列表
        """
        pass

    @abstractmethod
    def add_document_to_collection(self, document: any, collection_name: str = None):
        """
        添加文档到检索集合

        Args:
            document: 文档对象
            collection_name: 可选，指定添加的 Collection 名称
        """
        pass

    @abstractmethod
    def remove_document_from_collection(self, document_id: str, collection_name: str = None):
        """
        从检索集合中移除文档

        Args:
            document_id: 文档ID
            collection_name: 可选，指定移除的 Collection 名称
        """
        pass
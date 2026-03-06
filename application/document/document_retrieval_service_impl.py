"""
文档检索服务 Milvus 实现
"""

from typing import List
from domain.document.service.document_retrieval_service import DocumentRetrievalService
from domain.document.value_object.retrieval_result import RetrievalResult
from infrastructure.vector.vector_store import MilvusVectorStore
from infrastructure.model.embeddings_manager import EmbeddingsManager


class MilvusDocumentRetrievalService(DocumentRetrievalService):
    """文档检索服务 Milvus 实现"""

    def __init__(self):
        self.vector_store = MilvusVectorStore()
        self.embeddings_manager = EmbeddingsManager()

    def retrieve_similar_documents(self, query: str, limit: int = 5, score_threshold: float = 0.5) -> List[RetrievalResult]:
        """
        根据查询文本检索相似文档

        Args:
            query: 查询文本
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值

        Returns:
            检索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embeddings_manager.generate_embedding(query)

            # 搜索相似向量
            search_results = self.vector_store.search_documents(
                query_embedding=query_embedding,
                limit=limit
            )

            # 转换为检索结果值对象
            retrieval_results = []
            for result in search_results:
                # 计算相似度分数（根据 metric_type 转换）
                score = self._convert_distance_to_score(result["distance"])

                if score >= score_threshold:
                    retrieval_results.append(
                        RetrievalResult(
                            document_id=result["id"],
                            content=result["content"],
                            metadata=result["metadata"],
                            similarity_score=score,
                            distance=result["distance"]
                        )
                    )

            return retrieval_results

        except Exception as e:
            raise RuntimeError(f"检索相似文档失败: {str(e)}")

    def retrieve_similar_documents_by_embedding(self, query_embedding: List[float], limit: int = 5, score_threshold: float = 0.5) -> List[RetrievalResult]:
        """
        根据查询向量检索相似文档

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值

        Returns:
            检索结果列表
        """
        try:
            # 搜索相似向量
            search_results = self.vector_store.search_documents(
                query_embedding=query_embedding,
                limit=limit
            )

            # 转换为检索结果值对象
            retrieval_results = []
            for result in search_results:
                # 计算相似度分数（根据 metric_type 转换）
                score = self._convert_distance_to_score(result["distance"])

                if score >= score_threshold:
                    retrieval_results.append(
                        RetrievalResult(
                            document_id=result["id"],
                            content=result["content"],
                            metadata=result["metadata"],
                            similarity_score=score,
                            distance=result["distance"]
                        )
                    )

            return retrieval_results

        except Exception as e:
            raise RuntimeError(f"根据向量检索相似文档失败: {str(e)}")

    def add_document_to_collection(self, document):
        """
        添加文档到检索集合
        """
        try:
            # 确保文档有嵌入向量
            if not document.embedding:
                document.embedding = self.embeddings_manager.generate_embedding(document.content)

            # 插入到 Milvus
            self.vector_store.insert_documents([{
                "id": str(document.id),
                "content": document.content,
                "metadata": document.metadata,
                "embedding": document.embedding
            }])

        except Exception as e:
            raise RuntimeError(f"添加文档到检索集合失败: {str(e)}")

    def remove_document_from_collection(self, document_id: str):
        """
        从检索集合中移除文档
        """
        # 目前 Milvus 实现中未直接支持删除单个文档，需要根据实际需要实现
        raise NotImplementedError("从检索集合中移除文档功能未实现")

    def _convert_distance_to_score(self, distance: float) -> float:
        """
        将距离转换为相似度分数

        Args:
            distance: L2 距离

        Returns:
            相似度分数（0-1）
        """
        # 使用 Sigmoid 函数将 L2 距离转换为相似度分数
        # 假设距离范围大致在 0-2 之间
        score = 1 / (1 + distance)
        return max(0, min(1, score))
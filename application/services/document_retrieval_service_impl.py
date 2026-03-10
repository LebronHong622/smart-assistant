"""
文档检索服务 Milvus 实现
"""

from typing import List
from domain.document.service.document_retrieval_service import DocumentRetrievalService
from domain.document.value_object.retrieval_result import RetrievalResult
from domain.shared.ports.vector_store_port import VectorStorePort
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from domain.shared.ports.logger_port import LoggerPort


class MilvusDocumentRetrievalService(DocumentRetrievalService):
    """文档检索服务 Milvus 实现"""

    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_generator: EmbeddingGeneratorPort,
        logger: LoggerPort,
        default_collection: str = None
    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.logger = logger
        self.default_collection = default_collection

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
        try:
            # 生成查询向量
            query_embedding = self.embedding_generator.generate_embedding(query)

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
                    # 如果没有 metadata 字段，使用空字典或其他可用字段构建
                    metadata = result.get("metadata", {})

                    # 如果 metadata 为空，尝试从其他字段构建
                    if not metadata:
                        # 将非 metadata 字段放入 metadata
                        metadata = {k: v for k, v in result.items()
                                   if k not in ["id", "distance", "content"]}

                    retrieval_results.append(
                        RetrievalResult(
                            document_id=result["id"],
                            content=result["content"],
                            metadata=metadata,
                            similarity_score=score,
                            distance=result["distance"]
                        )
                    )

            return retrieval_results

        except Exception as e:
            self.logger.error(f"检索相似文档失败: {str(e)}")
            raise RuntimeError(f"检索相似文档失败: {str(e)}")

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
                    # 如果没有 metadata 字段，使用空字典或其他可用字段构建
                    metadata = result.get("metadata", {})

                    # 如果 metadata 为空，尝试从其他字段构建
                    if not metadata:
                        # 将非 metadata 字段放入 metadata
                        metadata = {k: v for k, v in result.items()
                                   if k not in ["id", "distance", "content"]}

                    retrieval_results.append(
                        RetrievalResult(
                            document_id=result["id"],
                            content=result["content"],
                            metadata=metadata,
                            similarity_score=score,
                            distance=result["distance"]
                        )
                    )

            return retrieval_results

        except Exception as e:
            self.logger.error(f"根据向量检索相似文档失败: {str(e)}")
            raise RuntimeError(f"根据向量检索相似文档失败: {str(e)}")

    def add_document_to_collection(self, document, collection_name: str = None):
        """
        添加文档到检索集合

        Args:
            document: 文档对象
            collection_name: 可选，指定添加的 Collection 名称
        """
        try:
            # 确保文档有嵌入向量
            if not document.embedding:
                document.embedding = self.embedding_generator.generate_embedding(document.content)

            # 获取 collection 的实际字段列表
            collection_fields = self.vector_store.get_collection_fields()

            # 转换文档为字典
            doc_dict = document.model_dump()

            # 过滤字段，只保留 collection 中存在的字段
            # 排除 Document 内部字段（id, metadata）
            filtered_data = {}
            for field_name in collection_fields:
                if field_name == "id":
                    # 跳过 auto_id 字段
                    continue
                elif field_name == "sparse_embedding" and field_name not in doc_dict:
                    # 稀疏向量填充空字典
                    filtered_data[field_name] = {}
                elif field_name in doc_dict:
                    filtered_data[field_name] = doc_dict[field_name]

            # 插入到 Milvus
            self.vector_store.insert_documents([filtered_data])

        except Exception as e:
            self.logger.error(f"添加文档到检索集合失败: {str(e)}")
            raise RuntimeError(f"添加文档到检索集合失败: {str(e)}")

    def remove_document_from_collection(self, document_id: str, collection_name: str = None):
        """
        从检索集合中移除文档

        Args:
            document_id: 文档ID
            collection_name: 可选，指定移除的 Collection 名称
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
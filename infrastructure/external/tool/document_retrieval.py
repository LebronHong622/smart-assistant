"""
文档检索工具
使用 LangChain 的 @tool 装饰器
"""

from langchain.tools import tool
from domain.document.service.document_retrieval_service import DocumentRetrievalService


@tool
def document_retrieval(query: str, limit: int = 5, score_threshold: float = 0.5, collection_name: str = None) -> dict:
    """
    文档检索工具，用于从 Milvus 向量数据库中检索与查询相关的文档。

    Args:
        query: 查询文本，要搜索的内容
        limit: 返回结果数量限制，默认 5
        score_threshold: 相似度分数阈值，默认 0.5（值越大，结果越相关）
        collection_name: 可选，指定检索的 Collection 名称，默认使用默认 Collection

    Returns:
        包含检索结果的字典，包括结果数量和每个文档的详细信息
    """
    try:
        # 初始化文档检索服务（延迟导入避免循环依赖）
        from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
        retrieval_service: DocumentRetrievalService = MilvusDocumentRetrievalService()

        # 检索相似文档
        results = retrieval_service.retrieve_similar_documents(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            collection_name=collection_name
        )

        # 格式化结果
        formatted_results = {
            "result_count": len(results),
            "results": []
        }

        for result in results:
            formatted_results["results"].append({
                "document_id": str(result.document_id),
                "content": result.content,
                "metadata": result.metadata,
                "similarity_score": round(result.similarity_score, 4),
                "distance": round(result.distance, 4)
            })

        return formatted_results

    except Exception as e:
        return {
            "error": str(e)
        }


@tool
def retrieve_similar_documents_by_embedding(embedding: list[float], limit: int = 5, score_threshold: float = 0.5, collection_name: str = None) -> dict:
    """
    根据嵌入向量检索相似文档

    Args:
        embedding: 查询嵌入向量，1536 维的浮点数组
        limit: 返回结果数量限制，默认 5
        score_threshold: 相似度分数阈值，默认 0.5
        collection_name: 可选，指定检索的 Collection 名称，默认使用默认 Collection

    Returns:
        包含检索结果的字典
    """
    try:
        # 初始化文档检索服务（延迟导入避免循环依赖）
        from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
        retrieval_service: DocumentRetrievalService = MilvusDocumentRetrievalService()

        # 检索相似文档
        results = retrieval_service.retrieve_similar_documents_by_embedding(
            query_embedding=embedding,
            limit=limit,
            score_threshold=score_threshold,
            collection_name=collection_name
        )

        # 格式化结果
        formatted_results = {
            "result_count": len(results),
            "results": []
        }

        for result in results:
            formatted_results["results"].append({
                "document_id": str(result.document_id),
                "content": result.content,
                "metadata": result.metadata,
                "similarity_score": round(result.similarity_score, 4),
                "distance": round(result.distance, 4)
            })

        return formatted_results

    except Exception as e:
        return {
            "error": str(e)
        }
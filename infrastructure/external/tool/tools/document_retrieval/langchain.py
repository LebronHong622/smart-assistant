"""
基于 LangChain 的文档检索工具
可与现有工具并行使用，符合 LangChain @tool 装饰器规范
"""
from typing import List, Dict, Optional
from langchain_core.tools import tool
from domain.entity.document.document import Document
from infrastructure.persistence.vector.repository.langchain_document_repository_impl import LangChainDocumentRepository
from infrastructure.core.log import app_logger
from config.settings import get_app_settings
from config.rag_settings import get_rag_settings
from infrastructure.external.tool.tools.document_retrieval.schema import LangChainDocumentRetrievalInput

settings = get_app_settings()
rag_settings = get_rag_settings()

# 模块级缓存：每个 collection_name 对应一个 LangChainDocumentRepository 单例
_repository_cache: Dict[str, LangChainDocumentRepository] = {}


def _get_repository(collection_name: Optional[str]) -> LangChainDocumentRepository:
    """
    获取或创建 LangChainDocumentRepository 单例实例

    Args:
        collection_name: 集合名称，如果为 None 则使用默认集合

    Returns:
        LangChainDocumentRepository 实例（单例）
    """
    if collection_name is None:
        collection_name = settings.milvus.milvus_collection_name

    if collection_name not in _repository_cache:
        app_logger.info(f"创建新的 LangChainDocumentRepository 实例，集合: {collection_name}")
        _repository_cache[collection_name] = LangChainDocumentRepository(
            collection_name=collection_name
        )

    return _repository_cache[collection_name]


@tool(
    "langchain_document_retrieval",
    description="基于 LangChain 的文档检索工具，用于检索与查询相关的文档内容。支持多集合隔离，每个集合名称维护单例 LangChainDocumentRepository 实例，提高重复查询性能。",
    args_schema=LangChainDocumentRetrievalInput
)
def langchain_document_retrieval(
    query: str,
    collection_name: Optional[str] = None
) -> List[Document]:
    """基于 LangChain 的文档检索工具"""
    try:
        app_logger.info(f"调用 LangChain 文档检索工具，查询: {query}, 集合: {collection_name}")

        # 获取单例仓库实例
        repository = _get_repository(collection_name)

        # 从配置读取默认参数进行检索
        results = repository.search_by_text(
            query=query,
            limit=rag_settings.retrieval.top_k,
            score_threshold=rag_settings.retrieval.score_threshold,
            with_score=rag_settings.retrieval.with_score
        )

        app_logger.info(f"LangChain 文档检索完成，找到 {len(results)} 条相关文档")
        return results

    except Exception as e:
        app_logger.error(f"LangChain 文档检索失败: {str(e)}")
        return []

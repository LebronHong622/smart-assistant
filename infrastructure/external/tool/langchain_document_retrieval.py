"""
基于 LangChain 的文档检索工具
可与现有工具并行使用，符合 LangChain @tool 装饰器规范
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from infrastructure.langchain import get_langchain_vector_store
from infrastructure.core.log import app_logger
from config.settings import AppSettings

settings = AppSettings()

@tool
def langchain_document_retrieval(
    query: str,
    top_k: int = 3,
    score_threshold: float = 0.8,
    provider: Optional[str] = None,
    embeddings: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    基于 LangChain 的文档检索工具，用于检索与查询相关的文档内容

    Args:
        query: 用户查询内容
        top_k: 返回最相关的文档数量，默认3条
        score_threshold: 相似度阈值，只有高于阈值的文档才会返回，默认0.8
        provider: 向量存储提供商，默认从配置中读取
        embeddings: 嵌入模型实例，默认使用系统配置的嵌入模型

    Returns:
        相关文档列表，每个文档包含content(内容)、metadata(元数据)、score(相似度)
    """
    try:
        app_logger.info(f"调用 LangChain 文档检索工具，查询: {query}, top_k: {top_k}, score_threshold: {score_threshold}")

        # 获取 LangChain VectorStore 实例
        vector_store = get_langchain_vector_store(provider=provider, embeddings=embeddings)

        # 带分数的相似性搜索
        results = vector_store.similarity_search_with_score(query, k=top_k)

        # 格式化结果
        formatted_results = []
        for doc, score in results:
            if score >= score_threshold:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })

        app_logger.info(f"LangChain 文档检索完成，找到 {len(formatted_results)} 条相关文档")
        return formatted_results

    except Exception as e:
        app_logger.error(f"LangChain 文档检索失败: {str(e)}")
        return []

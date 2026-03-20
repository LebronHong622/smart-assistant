"""
Milvus 向量存储实现
智能检测 collection 是否存在，支持混合搜索（BM25 + 密集向量）
"""
from typing import Any, Dict, Optional
from pymilvus import connections, utility
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus, BM25BuiltInFunction
from config.rag_settings import rag_settings
from infrastructure.core.log import app_logger


def check_collection_exists(connection_args: Dict, collection_name: str) -> bool:
    """检测 collection 是否存在

    Args:
        connection_args: Milvus 连接参数
        collection_name: Collection 名称

    Returns:
        bool: collection 是否存在
    """
    try:
        # 确保 Milvus 连接
        uri = connection_args.get("uri", "")
        connections.connect(alias="_check_conn", uri=uri)

        # 检查 collection 是否存在
        exists = collection_name in utility.list_collections(using="_check_conn")

        # 断开临时连接
        connections.disconnect(alias="_check_conn")

        return exists
    except Exception as e:
        app_logger.warning(f"检测 collection 存在性失败: {e}，假定不存在")
        return False


def build_bm25_function(
    config: "config.rag_settings.BM25FunctionConfig"
) -> "BM25BuiltInFunction":
    """根据配置构建 BM25BuiltInFunction

    Args:
        config: BM25 函数配置

    Returns:
        BM25BuiltInFunction 实例
    """
    from langchain_milvus import BM25BuiltInFunction

    if not config.enabled:
        return None

    # 构建分词器参数
    analyzer_params = None
    multi_analyzer_params = None

    if config.analyzer_params:
        analyzer_params = config.analyzer_params.model_dump(exclude_none=True)
    elif config.multi_analyzer_params:
        multi_analyzer_params = config.multi_analyzer_params.model_dump(exclude_none=True)

    bm25_function = BM25BuiltInFunction(
        input_field_names=config.input_field_names,
        output_field_names=config.output_field_names,
        enable_match=config.enable_match,
        function_name=config.function_name,
        analyzer_params=analyzer_params,
        multi_analyzer_params=multi_analyzer_params,
    )

    app_logger.info(
        f"创建 BM25 函数: input={config.input_field_names}, "
        f"output={config.output_field_names}, "
        f"function_name={config.function_name or 'auto'}"
    )

    return bm25_function


def create_milvus_store(
    embedding: Embeddings,
    collection_name: str,
    config: Optional[Any] = None,
    **kwargs,
) -> VectorStore:
    """创建 Milvus 向量存储

    智能检测 collection 是否存在：
    - 已存在：不传入 builtin_function，避免 schema 冲突
    - 不存在：根据配置创建 BM25 函数
    """
    from langchain_milvus import Milvus

    config = config or rag_settings.vector.milvus
    connection_args = {"uri": config.get_connection_uri()}

    langchain_config = config.langchain_config

    milvus_params = {
        "embedding_function": embedding,
        "collection_name": collection_name,
        "connection_args": connection_args,
        "auto_id": langchain_config.auto_id,
        "vector_field": langchain_config.vector_field,
        "text_field": langchain_config.text_field,
        "metadata_field": langchain_config.metadata_field,
        "enable_dynamic_field": config.enable_dynamic_field,
    }

    # 检测 collection 是否存在
    collection_exists = check_collection_exists(connection_args, collection_name)

    if langchain_config.enable_hybrid_search:
        # 创建 BM25 函数用于混合搜索
        bm25_function = build_bm25_function(langchain_config.bm25_function)
        milvus_params["builtin_function"] = bm25_function
        milvus_params["vector_field"] = [
            langchain_config.dense_vector_field,
            langchain_config.sparse_vector_field
        ]
        milvus_params["consistency_level"] = "Strong"

        if collection_exists:
            app_logger.info(
                f"Collection '{collection_name}' 已存在，使用 BM25 函数匹配现有 schema"
            )
        else:
            app_logger.info(f"Collection '{collection_name}' 不存在，创建 BM25 函数")

    return Milvus(**milvus_params, **kwargs)

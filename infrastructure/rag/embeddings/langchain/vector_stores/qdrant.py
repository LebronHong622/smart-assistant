"""
Qdrant 向量存储实现
"""
from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import Qdrant
from config.rag_settings import rag_settings


def create_qdrant_store(
    embedding: Embeddings,
    collection_name: str,
    config: Optional[Any] = None,
    **kwargs,
) -> VectorStore:
    """创建 Qdrant 向量存储"""
    from config.rag_settings import QdrantConfig

    config = config or rag_settings.vector.qdrant

    return Qdrant(
        embeddings=embedding,
        collection_name=collection_name,
        url=config.get_url(),
        api_key=config.connection.api_key or None,
        **kwargs,
    )

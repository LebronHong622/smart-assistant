"""
FAISS 向量存储实现
"""
from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from config.rag_settings import rag_settings


def create_faiss_store(
    embedding: Embeddings,
    index_name: str = "faiss_index",
    config: Optional[Any] = None,
    **kwargs,
) -> VectorStore:
    """创建 FAISS 向量存储"""
    config = config or rag_settings.vector.faiss

    return FAISS(
        embedding_function=embedding,
        index_name=index_name,
        **kwargs,
    )

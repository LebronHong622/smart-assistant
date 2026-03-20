"""
Chroma 向量存储实现
"""
from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from config.rag_settings import rag_settings


def create_chroma_store(
    embedding: Embeddings,
    collection_name: str,
    config: Optional[Any] = None,
    **kwargs,
) -> VectorStore:
    """创建 Chroma 向量存储"""
    from config.rag_settings import ChromaConfig

    config = config or rag_settings.vector.chroma

    return Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=config.persist_directory,
        client_settings=config.settings if config.settings else None,
        **kwargs,
    )

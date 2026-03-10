"""
LangChain 适配层
提供 LangChain 兼容的组件和接口
"""
from .factories.langchain_component_factory import (
    LangChainComponentFactory,
    langchain_component_factory,
    get_langchain_embeddings,
    get_langchain_vector_store
)

__all__ = [
    "LangChainComponentFactory",
    "langchain_component_factory",
    "get_langchain_embeddings",
    "get_langchain_vector_store"
]

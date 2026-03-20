"""
LangChain RAG 组件工厂（兼容性重导出
⚠️  此文件保留用于向后兼容
新代码请从新位置导入：
- LangChainLoaderFactory: infrastructure.rag.document_loader.langchain.loader
- LangChainEmbeddingAdapter: infrastructure.rag.embeddings.langchain.embedding_adapter
- LangChainEmbeddingFactory: infrastructure.rag.embeddings.langchain.embedding_factory
- LangChainVectorStoreFactory: infrastructure.rag.embeddings.langchain.vector_store
- LangChainSplitterFactory: infrastructure.rag.text_splitter.langchain.splitter
"""

# 向后兼容性重导出：所有类保持原样从新位置重导出
from infrastructure.rag.document_loader.langchain.loader import LangChainLoaderFactory
from infrastructure.rag.document_loader.langchain.adapters import (
    LoaderConfigAdapter,
    _create_json_metadata_func,
    _json_loader_adapter,
    _LOADER_ADAPTERS,
)
from infrastructure.rag.embeddings.langchain.embedding_adapter import LangChainEmbeddingAdapter
from infrastructure.rag.embeddings.langchain.embedding_factory import LangChainEmbeddingFactory
from infrastructure.rag.embeddings.langchain.vector_store import LangChainVectorStoreFactory
from infrastructure.rag.text_splitter.langchain.splitter import LangChainSplitterFactory
from infrastructure.rag.shared.converters import (
    convert_lc_to_domain as _convert_lc_to_domain,
    convert_domain_to_lc as _convert_domain_to_lc,
)

# 保持原有私有函数名称可用
_convert_lc_to_domain = _convert_lc_to_domain
_convert_domain_to_lc = _convert_domain_to_lc

__all__ = [
    "LangChainLoaderFactory",
    "LoaderConfigAdapter",
    "_create_json_metadata_func",
    "_json_loader_adapter",
    "_LOADER_ADAPTERS",
    "_convert_lc_to_domain",
    "_convert_domain_to_lc",
    "LangChainEmbeddingAdapter",
    "LangChainEmbeddingFactory",
    "LangChainVectorStoreFactory",
    "LangChainSplitterFactory",
]

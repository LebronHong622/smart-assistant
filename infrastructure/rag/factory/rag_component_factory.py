"""
RAG 组件工厂提供者
根据 framework 配置返回对应的框架实现
"""

from typing import TYPE_CHECKING, Optional
from domain.shared.enums import Framework
from config.settings import settings
from infrastructure.core.log import app_logger
from domain.shared.ports import (
    LoaderFactoryPort,
    EmbeddingFactoryPort,
    VectorStoreFactoryPort,
    SplitterFactoryPort,
)

if TYPE_CHECKING:
    from infrastructure.rag.document_loader.langchain.loader import LangChainLoaderFactory
    from infrastructure.rag.embeddings.langchain.embedding_factory import LangChainEmbeddingFactory
    from infrastructure.rag.embeddings.langchain.vector_store import LangChainVectorStoreFactory
    from infrastructure.rag.text_splitter.langchain.splitter import LangChainSplitterFactory


class RAGComponentFactory:
    """
    RAG 组件工厂提供者
    根据 settings.app.framework 配置返回对应框架的实现
    """

    _loader_factory: Optional[LoaderFactoryPort] = None
    _embedding_factory: Optional[EmbeddingFactoryPort] = None
    _vector_store_factory: Optional[VectorStoreFactoryPort] = None
    _splitter_factory: Optional[SplitterFactoryPort] = None

    @classmethod
    def _get_framework(cls) -> Framework:
        """获取当前框架配置"""
        return Framework(settings.app.framework)

    @classmethod
    def get_loader_factory(cls) -> LoaderFactoryPort:
        """获取文档加载器工厂"""
        if cls._loader_factory is None:
            framework = cls._get_framework()
            app_logger.info(f"初始化文档加载器工厂，框架: {framework.value}")

            if framework == Framework.LANGCHAIN:
                from infrastructure.rag.document_loader.langchain.loader import LangChainLoaderFactory
                cls._loader_factory = LangChainLoaderFactory
            elif framework == Framework.LLAMA_INDEX:
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == Framework.NATIVE:
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._loader_factory

    @classmethod
    def get_embedding_factory(cls) -> EmbeddingFactoryPort:
        """获取嵌入函数工厂"""
        if cls._embedding_factory is None:
            framework = cls._get_framework()
            app_logger.info(f"初始化嵌入函数工厂，框架: {framework.value}")

            if framework == Framework.LANGCHAIN:
                from infrastructure.rag.embeddings.langchain.embedding_factory import LangChainEmbeddingFactory
                cls._embedding_factory = LangChainEmbeddingFactory
            elif framework == Framework.LLAMA_INDEX:
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == Framework.NATIVE:
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._embedding_factory

    @classmethod
    def get_vector_store_factory(cls) -> VectorStoreFactoryPort:
        """获取向量存储工厂"""
        if cls._vector_store_factory is None:
            framework = cls._get_framework()
            app_logger.info(f"初始化向量存储工厂，框架: {framework.value}")

            if framework == Framework.LANGCHAIN:
                from infrastructure.rag.embeddings.langchain.vector_store import LangChainVectorStoreFactory
                cls._vector_store_factory = LangChainVectorStoreFactory
            elif framework == Framework.LLAMA_INDEX:
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == Framework.NATIVE:
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._vector_store_factory

    @classmethod
    def get_splitter_factory(cls) -> SplitterFactoryPort:
        """获取文本分割器工厂"""
        if cls._splitter_factory is None:
            framework = cls._get_framework()
            app_logger.info(f"初始化文本分割器工厂，框架: {framework.value}")

            if framework == Framework.LANGCHAIN:
                from infrastructure.rag.text_splitter.langchain.splitter import LangChainSplitterFactory
                cls._splitter_factory = LangChainSplitterFactory
            elif framework == Framework.LLAMA_INDEX:
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == Framework.NATIVE:
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._splitter_factory

    @classmethod
    def reset(cls) -> None:
        """重置工厂实例（用于测试）"""
        cls._loader_factory = None
        cls._embedding_factory = None
        cls._vector_store_factory = None
        cls._splitter_factory = None

"""
RAG 组件工厂提供者
根据 framework 配置返回对应的框架实现
"""

from typing import TYPE_CHECKING, Optional
from config.settings import settings
from infrastructure.core.log import app_logger
from infrastructure.rag.base import (
    ILoaderFactory,
    IEmbeddingFactory,
    IVectorStoreFactory,
    ISplitterFactory,
)

if TYPE_CHECKING:
    from infrastructure.rag.factory.langchain_factory import (
        LangChainLoaderFactory,
        LangChainEmbeddingFactory,
        LangChainVectorStoreFactory,
        LangChainSplitterFactory,
    )


class RAGComponentFactory:
    """
    RAG 组件工厂提供者
    根据 settings.app.framework 配置返回对应框架的实现
    """

    _loader_factory: Optional[ILoaderFactory] = None
    _embedding_factory: Optional[IEmbeddingFactory] = None
    _vector_store_factory: Optional[IVectorStoreFactory] = None
    _splitter_factory: Optional[ISplitterFactory] = None

    @classmethod
    def get_loader_factory(cls) -> ILoaderFactory:
        """获取文档加载器工厂"""
        if cls._loader_factory is None:
            framework = settings.app.framework
            app_logger.info(f"初始化文档加载器工厂，框架: {framework}")

            if framework == "langchain":
                from infrastructure.rag.factory.langchain_factory import LangChainLoaderFactory
                cls._loader_factory = LangChainLoaderFactory
            elif framework == "llamaindex":
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == "native":
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._loader_factory

    @classmethod
    def get_embedding_factory(cls) -> IEmbeddingFactory:
        """获取嵌入函数工厂"""
        if cls._embedding_factory is None:
            framework = settings.app.framework
            app_logger.info(f"初始化嵌入函数工厂，框架: {framework}")

            if framework == "langchain":
                from infrastructure.rag.factory.langchain_factory import LangChainEmbeddingFactory
                cls._embedding_factory = LangChainEmbeddingFactory
            elif framework == "llamaindex":
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == "native":
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._embedding_factory

    @classmethod
    def get_vector_store_factory(cls) -> IVectorStoreFactory:
        """获取向量存储工厂"""
        if cls._vector_store_factory is None:
            framework = settings.app.framework
            app_logger.info(f"初始化向量存储工厂，框架: {framework}")

            if framework == "langchain":
                from infrastructure.rag.factory.langchain_factory import LangChainVectorStoreFactory
                cls._vector_store_factory = LangChainVectorStoreFactory
            elif framework == "llamaindex":
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == "native":
                raise NotImplementedError(f"Native 框架暂未实现")
            else:
                raise ValueError(f"不支持的框架类型: {framework}")

        return cls._vector_store_factory

    @classmethod
    def get_splitter_factory(cls) -> ISplitterFactory:
        """获取文本分割器工厂"""
        if cls._splitter_factory is None:
            framework = settings.app.framework
            app_logger.info(f"初始化文本分割器工厂，框架: {framework}")

            if framework == "langchain":
                from infrastructure.rag.factory.langchain_factory import LangChainSplitterFactory
                cls._splitter_factory = LangChainSplitterFactory
            elif framework == "llamaindex":
                raise NotImplementedError(f"LlamaIndex 框架暂未实现")
            elif framework == "native":
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

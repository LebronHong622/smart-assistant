"""
LangChain RAG 组件工厂实现
基于 LangChain 框架的 RAG 组件工厂
"""

from typing import Any, Dict, List, Optional, Type
from importlib import import_module

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter

from config.rag_settings import rag_settings
from config.settings import settings
from domain.shared.ports import (
    LoaderFactoryPort,
    EmbeddingFactoryPort,
    VectorStoreFactoryPort,
    SplitterFactoryPort,
)
from infrastructure.core.log import app_logger


class LangChainLoaderFactory(LoaderFactoryPort):
    """
    LangChain 文档加载器工厂
    """

    _builtin_loaders: Dict[str, Type[Any]] = {}
    _custom_loaders: Dict[str, Type[Any]] = {}

    @classmethod
    def _ensure_builtin_loaders(cls):
        """延迟加载内置加载器"""
        if not cls._builtin_loaders:
            from langchain_community.document_loaders import (
                PyPDFLoader,
                TextLoader,
                CSVLoader,
                JSONLoader,
                DirectoryLoader,
                WebBaseLoader,
                UnstructuredMarkdownLoader,
                UnstructuredImageLoader,
            )
            cls._builtin_loaders = {
                "pdf": PyPDFLoader,
                "text": TextLoader,
                "csv": CSVLoader,
                "json": JSONLoader,
                "directory": DirectoryLoader,
                "web": WebBaseLoader,
                "markdown": UnstructuredMarkdownLoader,
                "image": UnstructuredImageLoader,
            }

    @classmethod
    def register_loader(cls, name: str, loader_class: Type[Any]) -> None:
        """注册自定义加载器"""
        cls._custom_loaders[name] = loader_class
        app_logger.info(f"注册自定义文档加载器: {name}")

    @classmethod
    def get_loader_class(cls, loader_type: str) -> Optional[Type[Any]]:
        """获取加载器类"""
        cls._ensure_builtin_loaders()

        if loader_type in cls._custom_loaders:
            return cls._custom_loaders[loader_type]

        if loader_type in cls._builtin_loaders:
            return cls._builtin_loaders[loader_type]

        config = rag_settings.get_loader_config(loader_type)
        if config and config.enabled:
            try:
                module = import_module(config.module)
                loader_class = getattr(module, config.class_name)
                cls._custom_loaders[loader_type] = loader_class
                return loader_class
            except (ImportError, AttributeError) as e:
                app_logger.error(f"加载文档加载器失败 [{loader_type}]: {e}")

        return None

    @classmethod
    def create_loader(
        cls,
        loader_type: str,
        file_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建文档加载器实例"""
        config = rag_settings.get_loader_config(loader_type)
        if not config or not config.enabled:
            raise ValueError(f"不支持的文档加载器类型: {loader_type}")

        loader_kwargs = {**config.config, **kwargs}
        loader_class = cls.get_loader_class(loader_type)
        if not loader_class:
            raise ValueError(f"找不到文档加载器类: {loader_type}")

        app_logger.info(f"创建文档加载器: {loader_type}, 文件: {file_path}")

        if file_path:
            return loader_class(file_path, **loader_kwargs)
        else:
            return loader_class(**loader_kwargs)

    @classmethod
    def load_documents(
        cls,
        loader_type: str,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """加载文档"""
        loader = cls.create_loader(loader_type, file_path, **kwargs)
        documents = loader.load()
        app_logger.info(f"加载文档完成: {file_path}, 共 {len(documents)} 个文档块")
        return documents

    @classmethod
    def load_from_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: str = "pdf",
        **kwargs
    ) -> List[Document]:
        """从目录加载文档"""
        from langchain_community.document_loaders import DirectoryLoader
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_kwargs=kwargs,
            show_progress=True
        )
        documents = loader.load()
        app_logger.info(f"从目录加载文档完成: {directory_path}, 共 {len(documents)} 个文档块")
        return documents

    @classmethod
    def list_supported_loaders(cls) -> List[str]:
        """列出所有支持的加载器类型"""
        cls._ensure_builtin_loaders()
        all_loaders = set(cls._builtin_loaders.keys())
        all_loaders.update(cls._custom_loaders.keys())
        all_loaders.update(rag_settings.loaders.keys())
        return list(all_loaders)


class LangChainEmbeddingFactory(EmbeddingFactoryPort):
    """
    LangChain 嵌入函数工厂
    """

    _registered_embeddings: Dict[str, Type[Embeddings]] = {}

    @classmethod
    def register_embedding(cls, name: str, embedding_class: Type[Embeddings]) -> None:
        """注册自定义嵌入函数"""
        cls._registered_embeddings[name] = embedding_class
        app_logger.info(f"注册嵌入函数: {name}")

    @classmethod
    def create_dashscope_embedding(
        cls,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Embeddings:
        """创建 DashScope 嵌入函数"""
        from langchain_community.embeddings import DashScopeEmbeddings

        config = rag_settings.get_embedding_config("dashscope")

        return DashScopeEmbeddings(
            model=model or (config.model if config else "text-embedding-v3"),
            dashscope_api_key=settings.dashscope.dashscope_api_key,
            batch_size=batch_size or (config.batch_size if config else 25),
        )

    @classmethod
    def create_openai_embedding(
        cls,
        model: str = "text-embedding-ada-002",
    ) -> Embeddings:
        """创建 OpenAI 嵌入函数"""
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model,
            openai_api_key=settings.api.deepseek_api_key,
            openai_api_base=settings.api.deepseek_api_base,
        )

    @classmethod
    def create_huggingface_embedding(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Embeddings:
        """创建 HuggingFace 嵌入函数"""
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)

    @classmethod
    def create_embedding(
        cls,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Embeddings:
        """创建嵌入函数实例"""
        provider = provider or rag_settings.rag_pipeline.default_embedding_model
        app_logger.info(f"创建嵌入函数: provider={provider}")

        if provider in cls._registered_embeddings:
            embedding_class = cls._registered_embeddings[provider]
            return embedding_class(**kwargs)

        creators = {
            "dashscope": cls.create_dashscope_embedding,
            "openai": cls.create_openai_embedding,
            "huggingface": cls.create_huggingface_embedding,
        }

        if provider not in creators:
            raise ValueError(f"不支持的嵌入函数类型: {provider}")

        return creators[provider](**kwargs)

    @classmethod
    def get_embedding_dimension(cls, provider: Optional[str] = None) -> int:
        """获取嵌入向量的维度"""
        provider = provider or rag_settings.rag_pipeline.default_embedding_model

        config = rag_settings.get_embedding_config(provider)
        if config:
            return config.dimension

        default_dimensions = {
            "dashscope": 1536,
            "openai": 1536,
            "huggingface": 384,
        }

        return default_dimensions.get(provider, 768)

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """列出所有支持的嵌入函数类型"""
        providers = {"dashscope", "openai", "huggingface"}
        providers.update(cls._registered_embeddings.keys())
        providers.update(rag_settings.embedding_models.keys())
        return list(providers)


class LangChainVectorStoreFactory(VectorStoreFactoryPort):
    """
    LangChain 向量存储工厂
    """

    _registered_stores: Dict[str, Type[VectorStore]] = {}

    @classmethod
    def register_store(cls, name: str, store_class: Type[VectorStore]) -> None:
        """注册自定义向量存储"""
        cls._registered_stores[name] = store_class
        app_logger.info(f"注册向量存储: {name}")

    @classmethod
    def create_milvus_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> VectorStore:
        """创建 Milvus 向量存储"""
        from langchain_milvus import Milvus, BM25BuiltInFunction
        from config.rag_settings import MilvusConfig

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
        }

        if langchain_config.enable_hybrid_search:
            app_logger.info("启用混合检索模式 (Dense + Sparse BM25)")
            milvus_params["builtin_function"] = BM25BuiltInFunction()
            milvus_params["vector_field"] = [
                langchain_config.dense_vector_field,
                langchain_config.sparse_vector_field
            ]
            milvus_params["consistency_level"] = "Strong"

        return Milvus(**milvus_params, **kwargs)

    @classmethod
    def create_chroma_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> VectorStore:
        """创建 Chroma 向量存储"""
        from langchain_chroma import Chroma
        from config.rag_settings import ChromaConfig

        config = config or rag_settings.vector.chroma

        return Chroma(
            embedding_function=embedding,
            collection_name=collection_name,
            persist_directory=config.persist_directory,
            client_settings=config.settings if config.settings else None,
            **kwargs,
        )

    @classmethod
    def create_faiss_store(
        cls,
        embedding: Embeddings,
        index_name: str = "faiss_index",
        config: Optional[Any] = None,
        **kwargs,
    ) -> VectorStore:
        """创建 FAISS 向量存储"""
        from langchain_community.vectorstores import FAISS

        config = config or rag_settings.vector.faiss

        return FAISS(
            embedding_function=embedding,
            index_name=index_name,
            **kwargs,
        )

    @classmethod
    def create_qdrant_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        config: Optional[Any] = None,
        **kwargs,
    ) -> VectorStore:
        """创建 Qdrant 向量存储"""
        from langchain_qdrant import Qdrant
        from config.rag_settings import QdrantConfig

        config = config or rag_settings.vector.qdrant

        return Qdrant(
            embeddings=embedding,
            collection_name=collection_name,
            url=config.get_url(),
            api_key=config.connection.api_key or None,
            **kwargs,
        )

    @classmethod
    def create_store(
        cls,
        embedding: Embeddings,
        collection_name: str,
        provider: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        """创建向量存储实例"""
        vector_config = rag_settings.get_vector_config()
        provider = provider or vector_config.provider
        app_logger.info(f"创建向量存储: provider={provider}, collection={collection_name}")

        if provider in cls._registered_stores:
            store_class = cls._registered_stores[provider]
            return store_class(
                embedding_function=embedding,
                collection_name=collection_name,
                **kwargs,
            )

        creators = {
            "milvus": cls.create_milvus_store,
            "chroma": cls.create_chroma_store,
            "faiss": cls.create_faiss_store,
            "qdrant": cls.create_qdrant_store,
        }

        if provider not in creators:
            raise ValueError(f"不支持的向量存储类型: {provider}")

        return creators[provider](
            embedding=embedding,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def get_store_config(cls, provider: Optional[str] = None) -> Any:
        """获取向量存储配置"""
        vector_config = rag_settings.get_vector_config()
        provider = provider or vector_config.provider
        return vector_config.get_active_config()

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """列出所有支持的向量存储类型"""
        providers = {"milvus", "chroma", "faiss", "qdrant"}
        providers.update(cls._registered_stores.keys())
        return list(providers)


class LangChainSplitterFactory(SplitterFactoryPort):
    """
    LangChain 文本分割器工厂
    """

    _builtin_splitters: Dict[str, Type[TextSplitter]] = {}
    _custom_splitters: Dict[str, Type[TextSplitter]] = {}

    @classmethod
    def _ensure_builtin_splitters(cls):
        """延迟加载内置分割器"""
        if not cls._builtin_splitters:
            from langchain_text_splitters import (
                RecursiveCharacterTextSplitter,
                CharacterTextSplitter,
                TokenTextSplitter,
                MarkdownTextSplitter,
            )
            cls._builtin_splitters = {
                "recursive": RecursiveCharacterTextSplitter,
                "character": CharacterTextSplitter,
                "token": TokenTextSplitter,
                "markdown": MarkdownTextSplitter,
            }

    @classmethod
    def register_splitter(cls, name: str, splitter_class: Type[TextSplitter]) -> None:
        """注册自定义分割器"""
        cls._custom_splitters[name] = splitter_class
        app_logger.info(f"注册自定义文本分割器: {name}")

    @classmethod
    def get_splitter_class(cls, splitter_type: str) -> Optional[Type[TextSplitter]]:
        """获取分割器类"""
        cls._ensure_builtin_splitters()

        if splitter_type in cls._custom_splitters:
            return cls._custom_splitters[splitter_type]

        if splitter_type in cls._builtin_splitters:
            return cls._builtin_splitters[splitter_type]

        config = rag_settings.get_splitter_config(splitter_type)
        if config and config.enabled:
            try:
                module = import_module(config.module)
                splitter_class = getattr(module, config.class_name)
                cls._custom_splitters[splitter_type] = splitter_class
                return splitter_class
            except (ImportError, AttributeError) as e:
                app_logger.error(f"加载文本分割器失败 [{splitter_type}]: {e}")

        return None

    @classmethod
    def create_splitter(
        cls,
        splitter_type: str = "recursive",
        **kwargs
    ) -> TextSplitter:
        """创建文本分割器实例"""
        config = rag_settings.get_splitter_config(splitter_type)
        if not config or not config.enabled:
            raise ValueError(f"不支持的文本分割器类型: {splitter_type}")

        splitter_kwargs = {**config.config, **kwargs}
        splitter_class = cls.get_splitter_class(splitter_type)
        if not splitter_class:
            raise ValueError(f"找不到文本分割器类: {splitter_type}")

        app_logger.info(f"创建文本分割器: {splitter_type}, 参数: {splitter_kwargs}")

        return splitter_class(**splitter_kwargs)

    @classmethod
    def split_documents(
        cls,
        documents: List[Document],
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[Document]:
        """分割文档"""
        splitter = cls.create_splitter(splitter_type, **kwargs)
        split_docs = splitter.split_documents(documents)
        app_logger.info(f"文档分割完成: 原始 {len(documents)} 个 -> 分割后 {len(split_docs)} 个")
        return split_docs

    @classmethod
    def split_text(
        cls,
        text: str,
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[str]:
        """分割文本"""
        splitter = cls.create_splitter(splitter_type, **kwargs)
        return splitter.split_text(text)

    @classmethod
    def create_code_splitter(
        cls,
        language: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> TextSplitter:
        """创建代码分割器"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @classmethod
    def list_supported_splitters(cls) -> List[str]:
        """列出所有支持的分割器类型"""
        cls._ensure_builtin_splitters()
        all_splitters = set(cls._builtin_splitters.keys())
        all_splitters.update(cls._custom_splitters.keys())
        all_splitters.update(rag_settings.splitters.keys())
        return list(all_splitters)

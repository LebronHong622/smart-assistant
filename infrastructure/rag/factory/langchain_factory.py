"""
LangChain RAG 组件工厂实现
基于 LangChain 框架的 RAG 组件工厂
内部进行领域 Document 与 LangChain Document 的转换
"""

from typing import Any, Dict, List, Optional, Type
from importlib import import_module
from pydantic import ConfigDict

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from pymilvus import connections, utility

from config.rag_settings import rag_settings
from config.settings import settings
from domain.document.entity.document import Document
from domain.shared.ports import (
    LoaderFactoryPort,
    EmbeddingFactoryPort,
    VectorStoreFactoryPort,
    SplitterFactoryPort,
)
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from infrastructure.core.log import app_logger


def _convert_lc_to_domain(lc_docs: List[LCDocument]) -> List[Document]:
    """将 LangChain Document 转换为领域 Document"""
    return [
        Document(content=doc.page_content, metadata=doc.metadata)
        for doc in lc_docs
    ]


def _convert_domain_to_lc(docs: List[Document]) -> List[LCDocument]:
    """将领域 Document 转换为 LangChain Document"""
    return [
        LCDocument(page_content=doc.content, metadata=doc.metadata or {})
        for doc in docs
    ]


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
        """加载文档，返回领域 Document"""
        loader = cls.create_loader(loader_type, file_path, **kwargs)
        lc_docs = loader.load()
        app_logger.info(f"加载文档完成: {file_path}, 共 {len(lc_docs)} 个文档块")
        # 转换为领域 Document
        return _convert_lc_to_domain(lc_docs)

    @classmethod
    def load_from_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: str = "pdf",
        **kwargs
    ) -> List[Document]:
        """从目录加载文档，返回领域 Document"""
        from langchain_community.document_loaders import DirectoryLoader
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_kwargs=kwargs,
            show_progress=True
        )
        lc_docs = loader.load()
        app_logger.info(f"从目录加载文档完成: {directory_path}, 共 {len(lc_docs)} 个文档块")
        # 转换为领域 Document
        return _convert_lc_to_domain(lc_docs)

    @classmethod
    def list_supported_loaders(cls) -> List[str]:
        """列出所有支持的加载器类型"""
        cls._ensure_builtin_loaders()
        all_loaders = set(cls._builtin_loaders.keys())
        all_loaders.update(cls._custom_loaders.keys())
        all_loaders.update(rag_settings.loaders.keys())
        return list(all_loaders)


class LangChainEmbeddingAdapter(EmbeddingGeneratorPort):
    """
    将 LangChain Embeddings 适配为领域 EmbeddingGeneratorPort

    实现领域接口，内部使用 LangChain Embeddings 进行实际的嵌入生成。
    """

    def __init__(self, embeddings: Embeddings, dimension: int = 768):
        self._embeddings = embeddings
        self._dimension = dimension

    # === 文本嵌入（同步）===

    def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        return self._embeddings.embed_documents(texts)

    # === 文档嵌入（同步）===

    def embed_document(self, document: Document) -> Document:
        """为单个文档生成嵌入向量，返回带嵌入的文档"""
        embedding = self.embed_text(document.content)
        document.embedding = embedding
        return document

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """批量生成文档的嵌入向量，返回带嵌入的文档列表"""
        texts = [doc.content for doc in documents]
        embeddings = self.embed_texts(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return documents

    # === 异步方法 ===

    async def aembed_text(self, text: str) -> List[float]:
        """异步生成单个文本的嵌入向量"""
        return await self._embeddings.aembed_query(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """异步批量生成文本的嵌入向量"""
        return await self._embeddings.aembed_documents(texts)

    async def aembed_document(self, document: Document) -> Document:
        """异步为单个文档生成嵌入向量"""
        embedding = await self.aembed_text(document.content)
        document.embedding = embedding
        return document

    async def aembed_documents(self, documents: List[Document]) -> List[Document]:
        """异步批量生成文档的嵌入向量"""
        texts = [doc.content for doc in documents]
        embeddings = await self.aembed_texts(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return documents

    # === 元信息 ===

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self._dimension

    # === LangChain 兼容接口（用于需要原始 Embeddings 的场景）===

    def to_langchain_embeddings(self) -> Embeddings:
        """获取原始 LangChain Embeddings 实例"""
        return self._embeddings


class LangChainEmbeddingFactory(EmbeddingFactoryPort):
    """
    LangChain 嵌入函数工厂

    创建 LangChain Embeddings 并包装为领域 EmbeddingGeneratorPort 接口。
    """

    _registered_embeddings: Dict[str, Type[Embeddings]] = {}

    @classmethod
    def register_embedding(cls, name: str, embedding_class: Type[Embeddings]) -> None:
        """注册自定义嵌入函数"""
        cls._registered_embeddings[name] = embedding_class
        app_logger.info(f"注册嵌入函数: {name}")

    @classmethod
    def _create_dashscope_embedding(
        cls,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Embeddings:
        """创建 DashScope 嵌入函数

        支持通过 dimension 参数指定输出向量维度（text-embedding-v3 支持 1024/768/512）
        """
        from langchain_community.embeddings import DashScopeEmbeddings

        config = rag_settings.get_embedding_config("dashscope")
        actual_model = model or (config.model if config else "text-embedding-v3")
        actual_dimension = dimension or (config.dimension if config else None)

        # 创建支持 dimension 参数的自定义 DashScope Embeddings
        class DashScopeEmbeddingsWithDimension(DashScopeEmbeddings):
            """支持 dimension 参数的 DashScope Embeddings"""
            dimension: Optional[int] = None

            model_config = ConfigDict(extra="allow")

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                kwargs = {"input": texts, "text_type": "document", "model": self.model}
                if self.dimension:
                    kwargs["dimension"] = self.dimension
                result = self.client.call(**kwargs)
                if result.status_code == 200:
                    return [item["embedding"] for item in result.output["embeddings"]]
                raise ValueError(f"Embedding failed: {result.message}")

            def embed_query(self, text: str) -> List[float]:
                kwargs = {"input": text, "text_type": "query", "model": self.model}
                if self.dimension:
                    kwargs["dimension"] = self.dimension
                result = self.client.call(**kwargs)
                if result.status_code == 200:
                    return result.output["embeddings"][0]["embedding"]
                raise ValueError(f"Embedding failed: {result.message}")

        return DashScopeEmbeddingsWithDimension(
            model=actual_model,
            dashscope_api_key=settings.dashscope.dashscope_api_key,
            dimension=actual_dimension,
        )

    @classmethod
    def _create_openai_embedding(
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
    def _create_huggingface_embedding(
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
    ) -> EmbeddingGeneratorPort:
        """创建嵌入函数实例，返回领域接口"""
        provider = provider or rag_settings.rag_pipeline.default_embedding_model
        app_logger.info(f"创建嵌入函数: provider={provider}")

        # 获取嵌入维度
        dimension = cls.get_embedding_dimension(provider)

        # 创建 LangChain Embeddings
        if provider in cls._registered_embeddings:
            embedding_class = cls._registered_embeddings[provider]
            lc_embeddings = embedding_class(**kwargs)
        else:
            creators = {
                "dashscope": cls._create_dashscope_embedding,
                "openai": cls._create_openai_embedding,
                "huggingface": cls._create_huggingface_embedding,
            }

            if provider not in creators:
                raise ValueError(f"不支持的嵌入函数类型: {provider}")

            lc_embeddings = creators[provider](**kwargs)

        # 包装为领域接口
        return LangChainEmbeddingAdapter(lc_embeddings, dimension=dimension)

    @classmethod
    def get_embedding_dimension(cls, provider: Optional[str] = None) -> int:
        """获取嵌入向量的维度"""
        provider = provider or rag_settings.rag_pipeline.default_embedding_model

        config = rag_settings.get_embedding_config(provider)
        if config:
            return config.dimension

        default_dimensions = {
            "dashscope": 768,
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
        """创建 Milvus 向量存储
        
        智能检测 collection 是否存在：
        - 已存在：不传入 builtin_function，避免 schema 冲突
        - 不存在：根据配置创建 BM25 函数
        """
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
            "enable_dynamic_field": config.enable_dynamic_field,
        }

        # 检测 collection 是否存在
        collection_exists = cls._check_collection_exists(connection_args, collection_name)

        if langchain_config.enable_hybrid_search:
            # 创建 BM25 函数用于混合搜索
            bm25_function = cls._build_bm25_function(langchain_config.bm25_function)
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

    @classmethod
    def _check_collection_exists(cls, connection_args: Dict, collection_name: str) -> bool:
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

    @classmethod
    def _build_bm25_function(
        cls,
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
        """分割文档，接受并返回领域 Document"""
        # 转换为 LangChain Document
        lc_docs = _convert_domain_to_lc(documents)
        
        splitter = cls.create_splitter(splitter_type, **kwargs)
        split_lc_docs = splitter.split_documents(lc_docs)
        app_logger.info(f"文档分割完成: 原始 {len(documents)} 个 -> 分割后 {len(split_lc_docs)} 个")
        
        # 转换回领域 Document
        return _convert_lc_to_domain(split_lc_docs)

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

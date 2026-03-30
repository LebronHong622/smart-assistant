"""
RAG 离线流程服务
整合文档加载、分块、向量存储的完整流程

⚠️  已废弃：请使用 domain.document.service.rag_processing_service.RAGProcessingService 替代
此类仅为向后兼容保留，将在未来版本中移除
"""
import warnings
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from domain.entity.document.document import Document as DomainDocument
from domain.repository.document.document_repository import DocumentRepository
from infrastructure.rag.document_loader.loader_factory import DocumentLoaderFactory
from infrastructure.rag.text_splitter.splitter_factory import TextSplitterFactory
from config.rag_settings import rag_settings
from infrastructure.core.log import app_logger


class RAGPipeline:
    """
    ⚠️  已废弃：请使用 RAGProcessingService 替代
    RAG 离线流程服务

    提供完整的 RAG 离线流程支持：
    - 文档加载（支持多种格式）
    - 文档分块（支持多种策略）
    - 向量存储（支持多业务领域）
    - 多模态文档支持
    """

    def __init__(
        self,
        embedding_function: Optional[Embeddings] = None,
        domain: Optional[str] = None,
        document_repository: Optional[DocumentRepository] = None,
    ):
        """
        初始化 RAG Pipeline

        Args:
            embedding_function: 嵌入函数
            domain: 业务领域（用于创建独立的 collection）
            document_repository: 文档仓库实例（依赖注入）
        """
        warnings.warn(
            "RAGPipeline 已废弃，请使用 RAGProcessingService 替代。"
            "未来版本将移除 RAGPipeline 类。",
            DeprecationWarning,
            stacklevel=2
        )

        # 延迟导入打破循环依赖
        from application.services.document.rag_processing_service_impl import (
            RAGProcessingServiceImpl,
        )

        # 内部使用新的 RAGProcessingService 实现
        self._service = RAGProcessingServiceImpl(
            embedding_function=embedding_function,
            domain=domain or "default",
            document_repository=document_repository
        )

        app_logger.info(f"初始化 RAGPipeline (已废弃): domain={self._service._domain}, collection={self._service._collection_name}")

    @property
    def collection_name(self) -> str:
        """获取当前 collection 名称"""
        return self._service._collection_name

    @property
    def domain(self) -> str:
        """获取当前业务领域"""
        return self._service._domain

    def set_embedding_function(self, embedding_function: Embeddings) -> None:
        """
        设置嵌入函数

        Args:
            embedding_function: 嵌入函数实例
        """
        self._service.set_embedding_function(embedding_function)

    def load_documents(
        self,
        file_path: str,
        loader_type: Optional[str] = None,
        **loader_kwargs,
    ) -> List[Document]:
        """
        加载文档

        Args:
            file_path: 文件路径
            loader_type: 加载器类型（默认从配置读取）
            **loader_kwargs: 加载器额外参数

        Returns:
            加载的文档列表
        """
        loader_type = loader_type or rag_settings.default_loader
        app_logger.info(f"加载文档: {file_path}, loader={loader_type}")

        documents = DocumentLoaderFactory.load_documents(
            loader_type=loader_type,
            file_path=file_path,
            **loader_kwargs,
        )

        return documents

    def split_documents(
        self,
        documents: List[Document],
        splitter_type: Optional[str] = None,
        **splitter_kwargs,
    ) -> List[Document]:
        """
        分块文档

        Args:
            documents: 待分块的文档列表
            splitter_type: 分块器类型（默认从配置读取）
            **splitter_kwargs: 分块器额外参数

        Returns:
            分块后的文档列表
        """
        splitter_type = splitter_type or rag_settings.default_splitter
        app_logger.info(f"分块文档: splitter={splitter_type}, 输入文档数={len(documents)}")

        split_docs = TextSplitterFactory.split_documents(
            documents=documents,
            splitter_type=splitter_type,
            **splitter_kwargs,
        )

        return split_docs

    def generate_embeddings(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None,
    ) -> List[DomainDocument]:
        """
        为文档生成嵌入向量

        Args:
            documents: LangChain 文档列表
            batch_size: 批处理大小

        Returns:
            领域文档实体列表（包含嵌入向量）
        """
        if self._service._embedding_function is None:
            raise ValueError("嵌入函数未设置，无法生成嵌入向量")

        batch_size = batch_size or rag_settings.rag_pipeline.batch_size
        app_logger.info(f"生成嵌入向量: 文档数={len(documents)}, batch_size={batch_size}")

        domain_documents = []

        # 批量处理
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]

            # 生成嵌入向量
            embeddings = self._service._embedding_function.embed_documents(texts)

            for doc, embedding in zip(batch, embeddings):
                domain_doc = DomainDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    embedding=embedding,
                )
                domain_documents.append(domain_doc)

        app_logger.info(f"嵌入向量生成完成: 共 {len(domain_documents)} 个文档")
        return domain_documents

    def store_documents(
        self,
        documents: List[DomainDocument],
    ) -> List[DomainDocument]:
        """
        存储文档到向量数据库

        Args:
            documents: 领域文档实体列表（需包含嵌入向量）

        Returns:
            存储后的文档列表
        """
        repository = self._service._get_document_repository()

        # 批量存储
        batch_size = rag_settings.rag_pipeline.batch_size
        stored_documents = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            stored = repository.save_all(batch)
            stored_documents.extend(stored)
            app_logger.debug(f"存储文档批次 {i//batch_size + 1}: {len(batch)} 个")

        app_logger.info(f"文档存储完成: 共 {len(stored_documents)} 个文档到 {self.collection_name}")
        return stored_documents

    def process_file(
        self,
        file_path: str,
        loader_type: Optional[str] = None,
        splitter_type: Optional[str] = None,
        **kwargs,
    ) -> List[DomainDocument]:
        """
        处理单个文件的完整流程

        包含：加载 -> 分块 -> 生成嵌入 -> 存储

        Args:
            file_path: 文件路径
            loader_type: 加载器类型
            splitter_type: 分块器类型
            **kwargs: 额外参数

        Returns:
            处理后的文档列表
        """
        app_logger.info(f"开始处理文件: {file_path}")

        # 1. 加载文档
        documents = self.load_documents(file_path, loader_type, **kwargs)

        # 2. 分块
        split_docs = self.split_documents(documents, splitter_type, **kwargs)

        # 3. 生成嵌入
        domain_docs = self.generate_embeddings(split_docs)

        # 4. 存储
        stored_docs = self.store_documents(domain_docs)

        app_logger.info(f"文件处理完成: {file_path}, 共 {len(stored_docs)} 个文档块")
        return stored_docs

    def process_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: Optional[str] = None,
        splitter_type: Optional[str] = None,
        **kwargs,
    ) -> List[DomainDocument]:
        """
        处理目录中的所有文件

        Args:
            directory_path: 目录路径
            glob_pattern: 文件匹配模式
            loader_type: 加载器类型
            splitter_type: 分块器类型
            **kwargs: 额外参数

        Returns:
            处理后的所有文档列表
        """
        app_logger.info(f"开始处理目录: {directory_path}")

        # 加载目录中的所有文档
        documents = DocumentLoaderFactory.load_from_directory(
            directory_path=directory_path,
            glob_pattern=glob_pattern,
            loader_type=loader_type or rag_settings.default_loader,
        )

        # 分块
        split_docs = self.split_documents(documents, splitter_type, **kwargs)

        # 生成嵌入
        domain_docs = self.generate_embeddings(split_docs)

        # 存储
        stored_docs = self.store_documents(domain_docs)

        app_logger.info(f"目录处理完成: {directory_path}, 共 {len(stored_docs)} 个文档块")
        return stored_docs

    def search(
        self,
        query: str,
        limit: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索文档

        Args:
            query: 查询文本
            limit: 返回结果数量
            filter_expr: 过滤表达式

        Returns:
            匹配的文档列表
        """
        if self._service._embedding_function is None:
            raise ValueError("嵌入函数未设置，无法进行搜索")

        repository = self._service._get_document_repository()

        documents = repository.search_by_text(
            query=query,
            limit=limit,
            filter_expr=filter_expr,
        )

        return [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

    def search_by_vector(
        self,
        embedding: List[float],
        limit: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        通过向量搜索文档

        Args:
            embedding: 查询向量
            limit: 返回结果数量
            filter_expr: 过滤表达式

        Returns:
            匹配的文档列表
        """
        repository = self._service._get_document_repository()

        documents = repository.search_by_vector(
            embedding=embedding,
            limit=limit,
            filter_expr=filter_expr,
        )

        return [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

    def get_retriever(self, **kwargs) -> Any:
        """
        获取 LangChain 检索器

        Args:
            **kwargs: 检索器参数

        Returns:
            LangChain 检索器实例
        """
        return self._service._document_repository.get_retriever(**kwargs)


class RAGPipelineFactory:
    """
    ⚠️  已废弃：请使用 RAGProcessingServiceFactoryImpl 替代
    RAG Pipeline 工厂类

    用于创建和管理多个业务领域的 RAG Pipeline 实例
    """

    _pipelines: Dict[str, 'RAGPipeline'] = {}

    @classmethod
    def get_pipeline(
        cls,
        domain: str,
        embedding_function: Optional[Embeddings] = None,
        document_repository: Optional[DocumentRepository] = None,
    ) -> 'RAGPipeline':
        """
        获取或创建指定领域的 Pipeline

        Args:
            domain: 业务领域
            embedding_function: 嵌入函数
            document_repository: 文档仓库实例（依赖注入）

        Returns:
            RAGPipeline 实例
        """
        warnings.warn(
            "RAGPipelineFactory 已废弃，请使用 RAGProcessingServiceFactoryImpl 替代。"
            "未来版本将移除 RAGPipelineFactory 类。",
            DeprecationWarning,
            stacklevel=2
        )

        if domain not in cls._pipelines:
            # 如果没有提供 document_repository，创建一个默认的
            if document_repository is None:
                from infrastructure.persistence.vector.repository.langchain_document_repository_impl import (
                    LangChainDocumentRepository,
                )
                from infrastructure.rag.embeddings import VectorStoreFactory

                if embedding_function is None:
                    raise ValueError("必须提供 embedding_function 或 document_repository")

                vector_store = VectorStoreFactory.create_store(
                    embedding=embedding_function,
                    collection_name=f"doc_{domain}",
                )
                document_repository = LangChainDocumentRepository(
                    collection_name=f"doc_{domain}",
                    embedding_function=embedding_function,
                    vector_store=vector_store,
                )

            cls._pipelines[domain] = RAGPipeline(
                embedding_function=embedding_function,
                domain=domain,
                document_repository=document_repository,
            )
        elif embedding_function:
            cls._pipelines[domain].set_embedding_function(embedding_function)

        return cls._pipelines[domain]

    @classmethod
    def list_domains(cls) -> List[str]:
        """列出所有已创建的领域"""
        return list(cls._pipelines.keys())

    @classmethod
    def remove_pipeline(cls, domain: str) -> None:
        """移除指定领域的 Pipeline"""
        if domain in cls._pipelines:
            del cls._pipelines[domain]

"""
应用层：RAG处理服务实现
完全解耦 LangChain 框架，使用领域层 Document 和 EmbeddingGeneratorPort
"""
from typing import List, Optional, Dict
from uuid import UUID

from domain.document.entity.document import Document
from domain.document.service.rag_processing_service import (
    RAGProcessingService,
    RAGProcessingServiceFactory
)
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from infrastructure.rag.document_loader.loader_factory import DocumentLoaderFactory
from infrastructure.rag.text_splitter.splitter_factory import TextSplitterFactory
from config.rag_settings import rag_settings
from infrastructure.core.log import app_logger


class RAGProcessingServiceImpl(RAGProcessingService):
    """
    RAG处理服务实现
    所有外部框架实现完全封装在基础设施层，应用层只使用领域 Document 和 EmbeddingGeneratorPort
    """

    def __init__(
        self,
        embedding_function: Optional[EmbeddingGeneratorPort] = None,
        domain: str = "default",
        document_repository: DocumentRepository = None,
    ):
        self._embedding_function = embedding_function
        self._domain = domain
        self._collection_name = rag_settings.get_collection_name(self._domain)

        if document_repository is None:
            raise ValueError("document_repository 必须提供，不能自动创建")
        self._document_repository = document_repository

        app_logger.info(f"初始化 RAG 处理服务: domain={self._domain}, collection={self._collection_name}")

    def set_embedding_function(self, embedding_function: EmbeddingGeneratorPort) -> None:
        """设置嵌入函数"""
        self._embedding_function = embedding_function
        if self._document_repository and hasattr(self._document_repository, 'set_embedding_function'):
            self._document_repository.set_embedding_function(embedding_function)

    def process_document(self, document: Document) -> Document:
        """处理单个文档"""
        app_logger.info(f"处理单个文档: id={document.id}, 长度={document.text_length}")

        # 分块（使用领域 Document）
        split_docs = TextSplitterFactory.split_documents(
            documents=[document],
            splitter_type=rag_settings.default_splitter
        )

        # 生成嵌入
        if self._embedding_function is None:
            raise ValueError("嵌入函数未设置，无法生成嵌入向量")

        # 使用领域接口直接生成文档嵌入
        processed_docs = self._embedding_function.embed_documents(split_docs)

        # 存储
        stored_docs = self.add_documents(processed_docs)

        # 返回第一个文档（如果是单个文档分块，返回第一个作为代表）
        return stored_docs[0] if stored_docs else document

    def batch_process_documents(self, documents: List[Document]) -> List[Document]:
        """批量处理多个文档"""
        app_logger.info(f"批量处理文档: 数量={len(documents)}")

        # 分块（使用领域 Document）
        split_docs = TextSplitterFactory.split_documents(
            documents=documents,
            splitter_type=rag_settings.default_splitter
        )

        # 生成嵌入
        if self._embedding_function is None:
            raise ValueError("嵌入函数未设置，无法生成嵌入向量")

        batch_size = rag_settings.rag_pipeline.batch_size

        # 分批处理，使用领域接口直接生成文档嵌入
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            self._embedding_function.embed_documents(batch)

        # 存储
        stored_docs = self.add_documents(split_docs)
        app_logger.info(f"批量处理完成: 共生成 {len(stored_docs)} 个文档块")
        return stored_docs

    def retrieve_similar(self, query: str, limit: int = 5, score_threshold: float = 0.7) -> List[Document]:
        """检索相似文档"""
        app_logger.info(f"检索相似文档: query={query[:50]}..., limit={limit}, threshold={score_threshold}")

        documents = self._document_repository.search_by_text(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )

        return documents

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量存储"""
        app_logger.info(f"添加文档到向量存储: 数量={len(documents)}")

        batch_size = rag_settings.rag_pipeline.batch_size
        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            stored = self._document_repository.save_all(batch)
            all_ids.extend([str(doc.id) for doc in stored])
            app_logger.debug(f"存储批次 {i//batch_size + 1}: {len(batch)} 个文档")

        app_logger.info(f"文档添加完成: 共 {len(all_ids)} 个文档")
        return all_ids

    def delete_documents(self, document_ids: List[str]) -> bool:
        """删除文档"""
        app_logger.info(f"删除文档: ids={document_ids}")

        try:
            self._document_repository.delete_all(document_ids)
            app_logger.info(f"文档删除成功: 共 {len(document_ids)} 个")
            return True
        except Exception as e:
            app_logger.error(f"文档删除失败: {str(e)}")
            return False

    def get_document(self, document_id: str) -> Optional[Document]:
        """获取文档"""
        try:
            return self._document_repository.find_by_id(UUID(document_id))
        except Exception as e:
            app_logger.error(f"获取文档失败: {str(e)}")
            return None

    # 扩展方法，兼容原有RAGPipeline的功能
    def process_file(self, file_path: str, loader_type: Optional[str] = None, **kwargs) -> List[Document]:
        """处理单个文件"""
        loader_type = loader_type or rag_settings.default_loader
        app_logger.info(f"处理文件: {file_path}, loader={loader_type}")

        # 加载文档（直接返回领域 Document）
        docs = DocumentLoaderFactory.load_documents(
            loader_type=loader_type,
            file_path=file_path,
            **kwargs
        )

        # 批量处理
        return self.batch_process_documents(docs)

    def process_directory(self, directory_path: str, glob_pattern: str = "**/*", **kwargs) -> List[Document]:
        """处理目录"""
        app_logger.info(f"处理目录: {directory_path}, pattern={glob_pattern}")

        # 加载目录文档（直接返回领域 Document）
        docs = DocumentLoaderFactory.load_from_directory(
            directory_path=directory_path,
            glob_pattern=glob_pattern,
            loader_type=rag_settings.default_loader
        )

        # 批量处理
        return self.batch_process_documents(docs)


class RAGProcessingServiceFactoryImpl(RAGProcessingServiceFactory):
    """
    RAG处理服务工厂实现
    管理多个业务领域的服务实例
    """

    _instances: Dict[str, RAGProcessingServiceImpl] = {}

    def create_service(self, domain: str = "default", embedding_function: Optional[EmbeddingGeneratorPort] = None, document_repository: Optional[DocumentRepository] = None, **kwargs) -> RAGProcessingService:
        """创建或获取指定领域的服务实例"""
        if domain not in self._instances:
            if document_repository is None:
                raise ValueError("创建新服务时必须提供 document_repository")
            self._instances[domain] = RAGProcessingServiceImpl(
                embedding_function=embedding_function,
                domain=domain,
                document_repository=document_repository,
                **kwargs
            )
        elif embedding_function:
            self._instances[domain].set_embedding_function(embedding_function)

        return self._instances[domain]

    @classmethod
    def list_domains(cls) -> List[str]:
        """列出所有已创建的领域"""
        return list(cls._instances.keys())

    @classmethod
    def remove_service(cls, domain: str) -> None:
        """移除指定领域的服务实例"""
        if domain in cls._instances:
            del cls._instances[domain]

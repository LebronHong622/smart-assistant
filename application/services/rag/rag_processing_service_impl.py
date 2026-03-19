"""
应用层：RAG处理服务实现
完全解耦 LangChain 框架，使用领域层 Document
嵌入生成委托给基础设施层的 DocumentRepository 实现
"""
from typing import List, Optional, Dict, Union

from domain.document.entity.document import Document
from domain.document.service.rag_processing_service import (
    RAGProcessingService,
    RAGProcessingServiceFactory
)
from domain.document.repository.document_repository import DocumentRepository
from infrastructure.rag.document_loader.loader_factory import DocumentLoaderFactory
from infrastructure.rag.text_splitter.splitter_factory import TextSplitterFactory
from config.rag_settings import rag_settings
from infrastructure.core.log import app_logger


class RAGProcessingServiceImpl(RAGProcessingService):
    """
    RAG处理服务实现
    所有外部框架实现完全封装在基础设施层，应用层只做流程编排
    嵌入生成由基础设施层 DocumentRepository 负责
    """

    def __init__(
        self,
        domain: str = "default",
        document_repository: DocumentRepository = None,
    ):
        self._domain = domain
        self._collection_name = rag_settings.get_collection_name(self._domain)

        if document_repository is None:
            raise ValueError("document_repository 必须提供，不能自动创建")
        self._document_repository = document_repository

        # 如果 document_repository 有 collection_name setter，尝试设置
        # 这支持后置设置 collection_name 的模式
        if hasattr(document_repository, 'collection_name') and hasattr(type(document_repository), 'collection_name') and isinstance(getattr(type(document_repository), 'collection_name'), property):
            # 检查是否存在 setter 方法
            prop = getattr(type(document_repository), 'collection_name')
            if hasattr(prop, 'setter'):
                setattr(document_repository, 'collection_name', self._collection_name)
                app_logger.info(f"已为 document_repository 设置 collection_name: {self._collection_name}")

        app_logger.info(f"初始化 RAG 处理服务: domain={self._domain}, collection={self._collection_name}")

    def process_document(self, document: Document) -> Document:
        """处理单个文档"""
        app_logger.info(f"处理单个文档: id={document.id}, 长度={document.text_length}")

        # 分块（使用领域 Document）
        split_docs = TextSplitterFactory.split_documents(
            documents=[document],
            splitter_type=rag_settings.default_splitter
        )

        # 存储 - 仓储会处理嵌入生成（如果需要）
        stored_docs = self.add_documents(split_docs)

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

        # 存储 - 仓储会处理嵌入生成（如果需要）
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
            # 将字符串ID转换为整数ID（因为Milvus使用自增整数主键）
            ids_to_delete: List[Union[int, str]] = [
                int(doc_id) if doc_id.isdigit() else doc_id
                for doc_id in document_ids
            ]
            self._document_repository.delete_all(ids_to_delete)
            app_logger.info(f"文档删除成功: 共 {len(ids_to_delete)} 个")
            return True
        except Exception as e:
            app_logger.error(f"文档删除失败: {str(e)}")
            return False

    def get_document(self, document_id: str) -> Optional[Document]:
        """获取文档"""
        try:
            doc_id = int(document_id)
            return self._document_repository.find_by_id(doc_id)
        except ValueError:
            app_logger.error(f"文档ID格式错误，必须为整数: {document_id}")
            return None
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
        
        # 打印docs内容用于调试
        app_logger.info(f"Loaded documents: {len(docs)} documents")
        for i, doc in enumerate(docs):
            app_logger.info(f"Doc {i+1}: content_length={len(doc.content)}, metadata={doc.metadata}")
            app_logger.debug(f"Doc {i+1} content preview: {doc.content[:200]}...")

        # 批量处理
        return self.batch_process_documents(docs)

    def process_directory(self, directory_path: str, loader_type: Optional[str] = None, glob_pattern: str = "**/*", **kwargs) -> List[Document]:
        """处理目录"""
        app_logger.info(f"处理目录: {directory_path}, pattern={glob_pattern}")
        loader_type = loader_type or rag_settings.default_loader

        # 加载目录文档（直接返回领域 Document）
        docs = DocumentLoaderFactory.load_from_directory(
            directory_path=directory_path,
            glob_pattern=glob_pattern,
            loader_type=loader_type
        )

        # 批量处理
        return self.batch_process_documents(docs)


class RAGProcessingServiceFactoryImpl(RAGProcessingServiceFactory):
    """
    RAG处理服务工厂实现
    管理多个业务领域的服务实例
    """

    _instances: Dict[str, RAGProcessingServiceImpl] = {}

    def create_service(self, domain: str = "default", document_repository: Optional[DocumentRepository] = None, **kwargs) -> RAGProcessingService:
        """创建或获取指定领域的服务实例"""
        if domain not in self._instances:
            if document_repository is None:
                raise ValueError("创建新服务时必须提供 document_repository")
            self._instances[domain] = RAGProcessingServiceImpl(
                domain=domain,
                document_repository=document_repository,
                **kwargs
            )

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

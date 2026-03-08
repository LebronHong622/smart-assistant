"""
DocumentService 接口实现
使用本地磁盘存储，专注于文档 CRUD 管理
"""
from typing import List, Optional
from uuid import UUID
from domain.document.entity.document import Document
from domain.document.service.document_service import DocumentService
from domain.document.value_object.document_metadata import DocumentMetadata
from domain.document.repository.document_repository import DocumentRepository
from infrastructure.config.settings import settings
from infrastructure.log import app_logger


class DocumentServiceImpl(DocumentService):
    """
    文档管理服务实现
    使用本地磁盘存储，专注于文档 CRUD 管理
    """

    def __init__(self, document_repository: DocumentRepository = None):
        self.document_repository = document_repository or self._get_default_repository()

    def _get_default_repository(self) -> DocumentRepository:
        """根据配置获取默认的文档仓库实现"""
        storage_type = settings.document_storage.document_storage_type

        if storage_type == "milvus":
            from infrastructure.vector.repository.document_repository_impl import MilvusDocumentRepository
            return MilvusDocumentRepository()
        else:  # 默认使用本地存储
            from infrastructure.document.repository.local_document_repository import LocalDocumentRepository
            return LocalDocumentRepository()

    def create_document(self, content: str, metadata: DocumentMetadata) -> Document:
        """创建文档（不生成 embedding）"""
        app_logger.info(f"创建文档: {metadata.title}")

        # 创建文档实体（不生成 embedding）
        document = Document(
            content=content,
            metadata=metadata.model_dump(),
            embedding=None
        )

        # 保存文档到本地磁盘
        return self.document_repository.save(document)

    def update_document(self, document_id: UUID, content: Optional[str] = None, metadata: Optional[DocumentMetadata] = None) -> Document:
        """更新文档"""
        app_logger.info(f"更新文档: {document_id}")

        # 获取现有文档
        document = self.document_repository.find_by_id(document_id)
        if not document:
            raise RuntimeError(f"文档不存在: {document_id}")

        # 更新内容
        if content is not None:
            document.content = content

        # 更新元数据
        if metadata is not None:
            document.metadata = metadata.model_dump()

        # 保存更新
        return self.document_repository.save(document)

    def delete_document(self, document_id: UUID) -> None:
        """删除文档"""
        app_logger.info(f"删除文档: {document_id}")
        self.document_repository.delete_by_id(document_id)

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """获取文档"""
        return self.document_repository.find_by_id(document_id)

    def list_documents(self, limit: int = 10, offset: int = 0) -> List[Document]:
        """列表获取文档"""
        app_logger.debug(f"获取文档列表: limit={limit}, offset={offset}")
        return self.document_repository.find_all(limit=limit, offset=offset)

    def search_documents_by_metadata(self, metadata: dict) -> List[Document]:
        """根据元数据搜索文档"""
        app_logger.debug(f"根据元数据搜索文档: {metadata}")

        # 获取所有文档并筛选
        all_documents = self.document_repository.find_all()

        filtered_docs = []
        for doc in all_documents:
            match = True
            for key, value in metadata.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break

            if match:
                filtered_docs.append(doc)

        app_logger.info(f"根据元数据搜索到 {len(filtered_docs)} 个文档")
        return filtered_docs

    def count_documents(self) -> int:
        """统计文档数量"""
        return self.document_repository.count()


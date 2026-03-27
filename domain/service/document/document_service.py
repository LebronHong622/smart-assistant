"""
文档管理服务接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from domain.entity.document.document import Document
from domain.vo.document.document_metadata import DocumentMetadata


class DocumentService(ABC):
    """文档管理服务接口"""

    @abstractmethod
    def create_document(self, content: str, metadata: DocumentMetadata) -> Document:
        """创建文档"""
        pass

    @abstractmethod
    def update_document(self, document_id: UUID, content: Optional[str] = None, metadata: Optional[DocumentMetadata] = None) -> Document:
        """更新文档"""
        pass

    @abstractmethod
    def delete_document(self, document_id: UUID) -> None:
        """删除文档"""
        pass

    @abstractmethod
    def get_document(self, document_id: UUID) -> Optional[Document]:
        """获取文档"""
        pass

    @abstractmethod
    def list_documents(self, limit: int = 10, offset: int = 0) -> List[Document]:
        """列表获取文档"""
        pass

    @abstractmethod
    def search_documents_by_metadata(self, metadata: dict) -> List[Document]:
        """根据元数据搜索文档"""
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """统计文档数量"""
        pass

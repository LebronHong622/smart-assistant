"""
文档操作接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from domain.document.entity.document import Document


class DocumentRepository(ABC):
    """文档操作仓库接口"""

    @abstractmethod
    def save(self, document: Document) -> Document:
        """保存文档"""
        pass

    @abstractmethod
    def save_all(self, documents: List[Document]) -> List[Document]:
        """保存多个文档"""
        pass

    @abstractmethod
    def find_by_id(self, document_id: UUID) -> Optional[Document]:
        """根据 ID 查找文档"""
        pass

    @abstractmethod
    def find_all(self, limit: int = 1000, offset: int = 0) -> List[Document]:
        """查找所有文档（支持分页）"""
        pass

    @abstractmethod
    def delete(self, document: Document) -> None:
        """删除文档"""
        pass

    @abstractmethod
    def delete_by_id(self, document_id: UUID) -> None:
        """根据 ID 删除文档"""
        pass

    @abstractmethod
    def count(self) -> int:
        """统计文档数量"""
        pass
"""
文档集合操作接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from domain.entity.document.document_collection import DocumentCollection


class DocumentCollectionRepository(ABC):
    """文档集合操作仓库接口"""

    @abstractmethod
    def save(self, collection: DocumentCollection) -> DocumentCollection:
        """保存文档集合"""
        pass

    @abstractmethod
    def find_by_id(self, collection_id: UUID) -> Optional[DocumentCollection]:
        """根据 ID 查找文档集合"""
        pass

    @abstractmethod
    def find_by_name(self, name: str) -> Optional[DocumentCollection]:
        """根据名称查找文档集合"""
        pass

    @abstractmethod
    def find_all(self) -> List[DocumentCollection]:
        """查找所有文档集合"""
        pass

    @abstractmethod
    def delete(self, collection: DocumentCollection) -> None:
        """删除文档集合"""
        pass

    @abstractmethod
    def delete_by_id(self, collection_id: UUID) -> None:
        """根据 ID 删除文档集合"""
        pass

    @abstractmethod
    def count(self) -> int:
        """统计文档集合数量"""
        pass

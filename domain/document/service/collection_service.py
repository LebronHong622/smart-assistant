"""
集合管理服务接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from domain.document.entity.document_collection import DocumentCollection


class CollectionService(ABC):
    """集合管理服务接口"""

    @abstractmethod
    def create_collection(self, name: str, description: Optional[str] = None) -> DocumentCollection:
        """创建文档集合"""
        pass

    @abstractmethod
    def delete_collection(self, collection_id: UUID) -> None:
        """删除文档集合"""
        pass

    @abstractmethod
    def get_collection(self, collection_id: UUID) -> Optional[DocumentCollection]:
        """获取文档集合"""
        pass

    @abstractmethod
    def get_collection_by_name(self, name: str) -> Optional[DocumentCollection]:
        """根据名称获取文档集合"""
        pass

    @abstractmethod
    def list_collections(self) -> List[DocumentCollection]:
        """获取所有文档集合列表"""
        pass

    @abstractmethod
    def count_collections(self) -> int:
        """统计文档集合数量"""
        pass

    @abstractmethod
    def get_collection_info(self, collection_id: UUID) -> dict:
        """获取集合详细信息"""
        pass

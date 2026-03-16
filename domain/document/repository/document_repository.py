"""
文档操作接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from domain.document.entity.document import Document


class DocumentRepository(ABC):
    """文档操作仓库接口"""

    @abstractmethod
    def save(self, document: Document, **kwargs) -> Document:
        """保存文档"""
        pass

    @abstractmethod
    def save_all(self, documents: List[Document], **kwargs) -> List[Document]:
        """保存多个文档"""
        pass

    @abstractmethod
    def find_by_id(self, document_id: int) -> Optional[Document]:
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
    def delete_by_id(self, document_id: Union[int, str]) -> None:
        """根据 ID 删除文档"""
        pass

    @abstractmethod
    def delete_all(self, document_ids: List[Union[int, str]]) -> None:
        """批量删除文档"""
        pass

    @abstractmethod
    def count(self) -> int:
        """统计文档数量"""
        pass

    @abstractmethod
    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """通过文本搜索相似文档"""
        pass

    @abstractmethod
    def search_by_vector(
        self,
        embedding: List[float],
        limit: int = 5,
        **kwargs
    ) -> List[Document]:
        """通过向量搜索相似文档"""
        pass
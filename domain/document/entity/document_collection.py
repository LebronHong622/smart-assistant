"""
文档集合实体
"""

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4, UUID
from domain.document.entity.document import Document


@dataclass
class DocumentCollection:
    """
    文档集合实体

    表示一组相关的文档集合
    """
    id: UUID
    name: str
    description: Optional[str] = None
    documents: List[Document] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: "2024-01-01 00:00:00")
    updated_at: str = field(default_factory=lambda: "2024-01-01 00:00:00")

    def __post_init__(self):
        if not self.id:
            self.id = uuid4()

    def add_document(self, document: Document) -> None:
        """添加文档到集合"""
        self.documents.append(document)

    def remove_document(self, document_id: UUID) -> None:
        """从集合中移除文档"""
        self.documents = [doc for doc in self.documents if doc.id != document_id]

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """获取文档"""
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        return None

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)

    @property
    def has_embeddings(self) -> bool:
        """检查是否所有文档都有嵌入向量"""
        if not self.documents:
            return False
        return all(doc.has_embedding for doc in self.documents)
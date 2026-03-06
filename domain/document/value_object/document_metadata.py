"""
文档元数据值对象
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DocumentType(Enum):
    """文档类型枚举"""
    TXT = "txt"
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


class DocumentSource(Enum):
    """文档来源枚举"""
    UPLOAD = "upload"
    WEB = "web"
    DATABASE = "database"
    OTHER = "other"


@dataclass(frozen=True)
class DocumentMetadata:
    """
    文档元数据值对象

    包含文档的元信息，如标题、来源、类型等
    """
    title: str
    source: DocumentSource
    document_type: DocumentType
    author: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    language: str = "zh"
    tags: list[str] = None

    def __post_init__(self):
        # 确保标签不为 None
        if self.tags is None:
            object.__setattr__(self, "tags", [])

    def has_tag(self, tag: str) -> bool:
        """检查是否包含指定标签"""
        return tag in self.tags

    def with_tag(self, tag: str):
        """添加标签（返回新对象）"""
        if self.has_tag(tag):
            return self

        new_tags = self.tags.copy()
        new_tags.append(tag)
        return DocumentMetadata(
            title=self.title,
            source=self.source,
            document_type=self.document_type,
            author=self.author,
            created_at=self.created_at,
            updated_at=self.updated_at,
            language=self.language,
            tags=new_tags
        )

    def without_tag(self, tag: str):
        """移除标签（返回新对象）"""
        if not self.has_tag(tag):
            return self

        new_tags = [t for t in self.tags if t != tag]
        return DocumentMetadata(
            title=self.title,
            source=self.source,
            document_type=self.document_type,
            author=self.author,
            created_at=self.created_at,
            updated_at=self.updated_at,
            language=self.language,
            tags=new_tags
        )
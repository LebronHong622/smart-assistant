"""
文档实体
"""

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4, UUID


@dataclass
class Document:
    """
    文档实体

    表示一个可被嵌入和检索的文档
    """
    id: UUID
    content: str
    metadata: Optional[dict] = None
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if not self.id:
            self.id = uuid4()

        if not self.metadata:
            self.metadata = {}

    @property
    def text_length(self) -> int:
        """获取文档内容长度"""
        return len(self.content)

    @property
    def has_embedding(self) -> bool:
        """检查是否有嵌入向量"""
        return self.embedding is not None and len(self.embedding) > 0

    def update_metadata(self, key: str, value: str):
        """更新文档元数据"""
        self.metadata[key] = value
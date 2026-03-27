"""
文档实体
"""

from pydantic import BaseModel, Field
from typing import Optional


class Document(BaseModel):
    """
    文档实体

    表示一个可被嵌入和检索的文档

    注意：使用自增ID模式，id 在插入前为 None，插入后由数据库分配
    """
    id: Optional[int] = None  # 自增ID，插入后由数据库分配
    content: str
    metadata: Optional[dict] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    distance: Optional[float] = None  # Milvus 返回的距离
    similarity_score: Optional[float] = None  # 转换后的相似度分数 (0-1)

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def model_post_init(self, __context):
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

"""
检索结果值对象
"""

from dataclasses import dataclass
from typing import Optional
from uuid import UUID


@dataclass(frozen=True)
class RetrievalResult:
    """
    检索结果值对象

    表示从向量数据库中检索到的文档结果
    """
    document_id: UUID
    content: str
    metadata: Optional[dict]
    similarity_score: float
    distance: float

    def __post_init__(self):
        # 验证相似度分数在合理范围内
        if self.similarity_score < 0 or self.similarity_score > 1:
            raise ValueError("相似度分数必须在 0 到 1 之间")

        # 验证距离值大于等于 0
        if self.distance < 0:
            raise ValueError("距离值必须大于等于 0")

    @property
    def is_highly_relevant(self) -> bool:
        """判断结果是否高度相关（相似度 > 0.8）"""
        return self.similarity_score > 0.8

    @property
    def is_relevant(self) -> bool:
        """判断结果是否相关（相似度 > 0.5）"""
        return self.similarity_score > 0.5
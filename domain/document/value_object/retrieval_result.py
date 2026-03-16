"""
检索结果值对象
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class RetrievalResult(BaseModel):
    """
    检索结果值对象

    表示从向量数据库中检索到的文档结果
    """
    document_id: str
    content: str
    metadata: Optional[dict]
    similarity_score: float
    distance: float

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @field_validator('similarity_score')
    def check_similarity_range(cls, v):
        if v < 0 or v > 1:
            raise ValueError("相似度分数必须在 0 到 1 之间")
        return v

    @field_validator('distance')
    def check_distance_non_negative(cls, v):
        if v < 0:
            raise ValueError("距离值必须大于等于 0")
        return v

    @property
    def is_highly_relevant(self) -> bool:
        """判断结果是否高度相关（相似度 > 0.8）"""
        return self.similarity_score > 0.8

    @property
    def is_relevant(self) -> bool:
        """判断结果是否相关（相似度 > 0.5）"""
        return self.similarity_score > 0.5
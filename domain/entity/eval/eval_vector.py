"""
向量元数据实体
存储向量的元数据信息，实际向量存储在Milvus
"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class EvalVector(BaseModel):
    """向量元数据实体

    存储评测向量的元数据，实际向量数据存储在Milvus：
    - 关联评测任务和数据集版本
    - 存储原始内容和附加元数据
    - 通过vector_id关联Milvus中的向量
    """
    id: Optional[int] = None
    """数据库自增ID"""

    vector_id: str
    """向量唯一ID"""

    milvus_id: Optional[str] = None
    """Milvus中的向量ID"""

    task_id: str
    """关联的评测任务ID"""

    dataset_id: str
    """关联的数据集ID"""

    dataset_version: str
    """关联的数据集版本"""

    record_id: str
    """原始数据集中的记录ID"""

    content: Optional[str] = None
    """原始内容（可选）"""

    meta_json: Optional[Dict] = None
    """附加元数据JSON"""

    embedding: Optional[list[float]] = None
    """向量嵌入（临时，仅在插入时使用，不存储到DB）"""

    create_time: datetime = Field(default_factory=datetime.now)
    """创建时间"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def has_embedding(self) -> bool:
        """检查是否有嵌入向量"""
        return self.embedding is not None and len(self.embedding) > 0

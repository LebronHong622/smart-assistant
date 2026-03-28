"""
评测结果实体
存储单个评测指标的结果，一经保存不可修改删除
"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field

from domain.vo.eval.metric_value import MetricValue


class EvalResult(BaseModel):
    """评测结果实体

    存储单个评测指标的计算结果，核心规则：
    - 只允许新增，不允许修改/删除
    - 保证评测结果的可信度和可追溯性
    """
    id: Optional[int] = None
    """数据库自增ID"""

    result_id: str
    """结果唯一ID"""

    task_id: str
    """关联的评测任务ID"""

    dataset_id: str
    """关联的数据集ID"""

    dataset_version: str
    """关联的数据集版本"""

    model_version: str
    """关联的模型版本"""

    metric_name: str
    """指标名称（如 recall@k, mrr, ndcg 等）"""

    metric_value: MetricValue
    """指标值（包含置信区间）"""

    details: Optional[Dict] = None
    """详细结果数据（JSON格式）"""

    create_time: datetime = Field(default_factory=datetime.now)
    """创建时间"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "result_id": self.result_id,
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "metric_name": self.metric_name,
            "value": self.metric_value.value,
            "confidence_lower": self.metric_value.confidence_lower,
            "confidence_upper": self.metric_value.confidence_upper,
            "details": self.details,
            "create_time": self.create_time.isoformat() if self.create_time else None
        }

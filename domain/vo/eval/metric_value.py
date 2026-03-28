"""
指标值对象
包含指标值和置信区间
"""
from pydantic import BaseModel
from typing import Optional


class MetricValue(BaseModel):
    """指标值对象

    存储评测指标的数值以及可选的置信区间
    """
    value: float
    """指标数值"""

    confidence_lower: Optional[float] = None
    """置信区间下限"""

    confidence_upper: Optional[float] = None
    """置信区间上限"""

    @property
    def has_confidence_interval(self) -> bool:
        """检查是否有置信区间"""
        return self.confidence_lower is not None and self.confidence_upper is not None

    def to_dict(self) -> dict:
        """转换为字典"""
        result = {
            "value": self.value,
        }
        if self.has_confidence_interval:
            result["confidence_lower"] = self.confidence_lower
            result["confidence_upper"] = self.confidence_upper
        return result

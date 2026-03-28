"""
数据集状态值对象
"""
from enum import Enum


class DatasetStatus(str, Enum):
    """数据集状态

    - ACTIVE: 当前生效版本
    - DEPRECATED: 已废弃（被新版本替换）
    """
    ACTIVE = "active"
    DEPRECATED = "deprecated"

    @property
    def is_active(self) -> bool:
        """检查是否为活跃状态"""
        return self == DatasetStatus.ACTIVE

    @property
    def is_deprecated(self) -> bool:
        """检查是否已废弃"""
        return self == DatasetStatus.DEPRECATED

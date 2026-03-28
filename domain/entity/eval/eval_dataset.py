"""
测试数据集实体
评测领域的聚合根，一经发布不可修改，只能新版本替换
"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field

from domain.vo.eval.version import Version
from domain.vo.eval.dataset_status import DatasetStatus


class EvalDataset(BaseModel):
    """测试数据集实体

    表示一个版本化的测试数据集，核心不变性规则：
    - 一经发布，不可修改
    - 修改必须生成新版本
    - 旧版本只能标记为废弃，不能删除
    """
    id: Optional[int] = None
    """数据库自增ID，插入前为 None"""

    dataset_id: str
    """数据集业务ID，同一逻辑数据集不同版本共享此ID"""

    dataset_name: str
    """数据集名称"""

    version: Version
    """版本号，主版本.次版本格式"""

    file_path: str
    """本地文件路径"""

    create_time: datetime = Field(default_factory=datetime.now)
    """创建时间"""

    creator: str
    """创建者"""

    status: DatasetStatus = DatasetStatus.ACTIVE
    """状态：active 生效，deprecated 已废弃"""

    metadata: Dict = Field(default_factory=dict)
    """额外元数据"""

    task_count: int = 0
    """测试用例数量"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def is_active(self) -> bool:
        """检查是否为活跃状态"""
        return self.status.is_active

    @property
    def is_deprecated(self) -> bool:
        """检查是否已废弃"""
        return self.status.is_deprecated

    def deprecate(self) -> None:
        """标记为废弃

        当新版本发布后，旧版本需要标记为废弃
        """
        self.status = DatasetStatus.DEPRECATED

    def version_string(self) -> str:
        """获取版本字符串"""
        return self.version.to_string()

    def validate_immutability(self) -> None:
        """验证不变性规则

        已存在的数据集（id 不为 None）不允许修改，必须生成新版本
        此方法在保存前被调用，违反规则则抛出异常
        """
        if self.id is not None:
            raise ValueError(
                "已发布的数据集不允许修改，请基于此版本创建新版本"
            )

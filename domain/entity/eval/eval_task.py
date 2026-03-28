"""
评测任务实体
管理单个评测任务的生命周期
"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class EvalTaskStatus(str, Enum):
    """评测任务状态"""
    PENDING = "pending"
    """等待执行"""
    RUNNING = "running"
    """执行中"""
    COMPLETED = "completed"
    """已完成"""
    FAILED = "failed"
    """执行失败"""


class EvalTask(BaseModel):
    """评测任务实体

    表示一次完整的评测任务，包含：
    - 使用哪个数据集版本
    - 使用哪个模型版本
    - 执行状态和时间
    """
    id: Optional[int] = None
    """数据库自增ID"""

    task_id: str
    """任务唯一ID"""

    task_name: str
    """任务名称"""

    model_version: str
    """被评测的模型版本"""

    dataset_id: str
    """使用的数据集ID"""

    dataset_version: str
    """使用的数据集版本"""

    status: EvalTaskStatus = EvalTaskStatus.PENDING
    """任务状态"""

    create_time: datetime = Field(default_factory=datetime.now)
    """创建时间"""

    start_time: Optional[datetime] = None
    """开始执行时间"""

    end_time: Optional[datetime] = None
    """结束执行时间"""

    parameters: Dict = Field(default_factory=dict)
    """评测参数"""

    creator: str
    """创建者"""

    error_message: Optional[str] = None
    """错误信息（失败时）"""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    @property
    def duration_seconds(self) -> Optional[float]:
        """获取执行耗时（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def mark_running(self) -> None:
        """标记为开始执行"""
        self.status = EvalTaskStatus.RUNNING
        self.start_time = datetime.now()

    def mark_completed(self) -> None:
        """标记为完成"""
        self.status = EvalTaskStatus.COMPLETED
        self.end_time = datetime.now()

    def mark_failed(self, error_message: str) -> None:
        """标记为失败"""
        self.status = EvalTaskStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message

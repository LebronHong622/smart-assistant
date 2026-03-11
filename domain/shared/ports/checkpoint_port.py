"""
检查点存储端口 - 定义框架无关的检查点存储接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CheckpointPort(ABC):
    """检查点存储接口 - 框架无关的统一接口"""

    @abstractmethod
    def save_checkpoint(self, thread_id: str, checkpoint_data: Dict[str, Any]) -> bool:
        """保存检查点数据"""
        pass

    @abstractmethod
    def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """获取检查点数据"""
        pass

    @abstractmethod
    def delete_checkpoint(self, thread_id: str) -> bool:
        """删除检查点数据"""
        pass

    @abstractmethod
    def list_threads(self) -> List[str]:
        """列出所有线程ID"""
        pass
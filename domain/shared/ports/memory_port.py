"""
内存管理端口 - 定义内存管理接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class MemoryPort(ABC):
    """内存管理接口"""

    @abstractmethod
    def get_saver(self) -> Any:
        """获取存储保存器"""
        pass

    @abstractmethod
    def get_thread_memory_config(self, thread_id: str) -> Dict[str, Any]:
        """获取线程内存配置"""
        pass

    @abstractmethod
    def get_overflow_memory_middleware(self, method: str = "trim") -> Any:
        """获取溢出内存中间件"""
        pass

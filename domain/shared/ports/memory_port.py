"""
内存管理端口 - 定义内存管理接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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

    @abstractmethod
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史记录"""
        pass

    @abstractmethod
    def add_user_message(self, session_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加用户消息到会话历史"""
        pass

    @abstractmethod
    def add_assistant_message(self, session_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加助手消息到会话历史"""
        pass

    @abstractmethod
    def clear_history(self, session_id: str) -> bool:
        """清空会话历史"""
        pass

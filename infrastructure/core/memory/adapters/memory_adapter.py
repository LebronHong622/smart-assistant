"""
内存管理适配器 - 实现内存管理端口
合并了 MemoryManager 的逻辑
"""

from typing import Any
from langgraph.checkpoint.memory import InMemorySaver
from domain.shared.ports.memory_port import MemoryPort
from infrastructure.core.memory.middle_ware import trim_messages, summarize_messages, delete_old_messages
from domain.shared.enums import OverflowMemoryMethod
from infrastructure.persistence.cache.storage_factory import create_storage_saver


class MemoryAdapter(MemoryPort):
    """内存管理适配器实现"""

    def __init__(self):
        self.saver = create_storage_saver()

    def get_saver(self) -> Any:
        """获取存储保存器"""
        return self.saver

    def get_thread_memory_config(self, thread_id: str) -> dict[str, Any]:
        """获取线程内存配置"""
        return {"configurable": {"thread_id": "thread_" + thread_id}}

    def get_overflow_memory_middleware(self, method: str = "trim") -> Any:
        """获取溢出内存中间件"""
        if method == OverflowMemoryMethod.DELETE.value:
            return delete_old_messages
        return trim_messages

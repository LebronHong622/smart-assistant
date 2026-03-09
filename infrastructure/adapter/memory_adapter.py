"""
内存管理适配器 - 实现内存管理端口
"""

from domain.port.memory_port import MemoryPort
from infrastructure.memory.memory_manager import MemoryManager


class MemoryAdapter(MemoryPort):
    """内存管理适配器实现"""

    def __init__(self):
        self._memory_manager = MemoryManager()

    def get_saver(self):
        """获取存储保存器"""
        return self._memory_manager.get_saver()

    def get_thread_memory_config(self, thread_id: str) -> dict[str, any]:
        """获取线程内存配置"""
        return self._memory_manager.get_thread_memory_config(thread_id)

    def get_overflow_memory_middleware(self, method: str = "trim"):
        """获取溢出内存中间件"""
        return self._memory_manager.get_overflow_memory_middleware(method)

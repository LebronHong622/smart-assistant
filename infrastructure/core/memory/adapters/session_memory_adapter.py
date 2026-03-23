"""
内存管理适配器 - 实现内存管理端口
合并了 MemoryManager 的逻辑
"""

from typing import Any, Dict, List, Optional
from langgraph.checkpoint.memory import InMemorySaver
from domain.shared.ports.memory_port import MemoryPort
from infrastructure.core.memory.middle_ware import trim_messages, summarize_messages, delete_old_messages
from domain.shared.enums import OverflowMemoryMethod
from infrastructure.persistence.cache.storage_factory import create_storage_adapter, create_storage_saver
from infrastructure.persistence.cache.adapters.langchain_checkpoint_adapter import LangChainCheckpointAdapter
from infrastructure.core.log.adapters.logger_adapter import LoggerAdapter


class SessionMemoryAdapter(MemoryPort):
    """会话内存管理适配器实现"""

    def __init__(self):
        # 使用统一的适配器接口
        self.storage_adapter = create_storage_adapter()
        # 保持向后兼容
        self.saver = create_storage_saver()
        # 初始化日志
        self.logger = LoggerAdapter()

    def get_saver(self) -> Any:
        """获取存储保存器（保持向后兼容）"""
        return self.saver

    def get_thread_memory_config(self, thread_id: str) -> dict[str, Any]:
        """获取线程内存配置"""
        return {"configurable": {"thread_id": "thread_" + thread_id}}

    def get_overflow_memory_middleware(self, method: str = "trim") -> Any:
        """获取溢出内存中间件"""
        if method == OverflowMemoryMethod.DELETE.value:
            return delete_old_messages
        return trim_messages

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史记录"""
        try:
            # 使用thread_id作为session_id获取检查点数据
            checkpoint_data = self.storage_adapter.get_checkpoint(session_id)
            
            if checkpoint_data and 'checkpoint' in checkpoint_data:
                checkpoint = checkpoint_data['checkpoint']
                # 从检查点中提取消息历史
                if isinstance(checkpoint, dict) and 'messages' in checkpoint:
                    return checkpoint['messages']
            return []
        except Exception as e:
            self.logger.error(f"获取历史记录失败: {str(e)}")
            return []

    def add_user_message(self, session_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加用户消息到会话历史"""
        return self._add_message(session_id, "user", message, metadata)

    def add_assistant_message(self, session_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加助手消息到会话历史"""
        return self._add_message(session_id, "assistant", message, metadata)

    def _add_message(self, session_id: str, role: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加消息到会话历史"""
        try:
            # 获取当前检查点数据
            checkpoint_data = self.storage_adapter.get_checkpoint(session_id)
            
            if not checkpoint_data:
                # 创建新的检查点
                checkpoint_data = {
                    "checkpoint_id": f"{session_id}_initial",
                    "checkpoint": {
                        "messages": []
                    },
                    "metadata": {},
                    "parent_checkpoint_id": None
                }
            
            # 添加新消息
            message_data = {
                "role": role,
                "content": message,
                "timestamp": metadata.get("timestamp") if metadata else None
            }
            
            if metadata:
                message_data.update(metadata)
            
            checkpoint_data["checkpoint"]["messages"].append(message_data)
            
            # 保存更新后的检查点
            return self.storage_adapter.save_checkpoint(session_id, checkpoint_data)
        except Exception as e:
            self.logger.error(f"添加消息失败: {str(e)}")
            return False

    def clear_history(self, session_id: str) -> bool:
        """清空会话历史"""
        try:
            return self.storage_adapter.delete_checkpoint(session_id)
        except Exception as e:
            self.logger.error(f"清空历史失败: {str(e)}")
            return False

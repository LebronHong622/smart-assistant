"""
LangChain检查点适配器 - 将LangChain的BaseCheckpointSaver适配到统一的CheckpointPort接口
"""

from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig
from domain.shared.ports.checkpoint_port import CheckpointPort
from langgraph.checkpoint.base import BaseCheckpointSaver 


class LangChainCheckpointAdapter(CheckpointPort):
    """LangChain检查点适配器"""

    def __init__(self, langchain_saver: BaseCheckpointSaver):
        """
        初始化适配器
        
        Args:
            langchain_saver: LangChain的BaseCheckpointSaver实例
        """
        self.langchain_saver = langchain_saver

    def _create_config(self, thread_id: str) -> RunnableConfig:
        """创建RunnableConfig"""
        return {"configurable": {"thread_id": thread_id}}

    def save_checkpoint(self, thread_id: str, checkpoint_data: Dict[str, Any]) -> bool:
        """
        保存检查点数据
        
        Args:
            thread_id: 线程ID
            checkpoint_data: 检查点数据
            
        Returns:
            bool: 是否保存成功
        """
        try:
            config = self._create_config(thread_id)
            
            checkpoint = checkpoint_data.get("checkpoint", {})
            metadata = checkpoint_data.get("metadata", {})
            new_versions = checkpoint_data.get("new_versions", {})
            
            # 使用put方法保存检查点
            self.langchain_saver.put(config, checkpoint, metadata, new_versions)
            return True
        except Exception as e:
            print(f"保存检查点失败: {e}")
            return False

    def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        获取检查点数据
        
        Args:
            thread_id: 线程ID
            
        Returns:
            Optional[Dict[str, Any]]: 检查点数据，如果不存在则返回None
        """
        try:
            config = self._create_config(thread_id)
            
            # 使用get方法获取检查点
            checkpoint = self.langchain_saver.get(config)
            
            if checkpoint:
                # 获取元组以获取metadata
                checkpoint_tuple = self.langchain_saver.get_tuple(config)
                return {
                    "checkpoint": checkpoint,
                    "metadata": checkpoint_tuple.metadata if checkpoint_tuple else {}
                }
            return None
        except Exception as e:
            print(f"获取检查点失败: {e}")
            return None

    def delete_checkpoint(self, thread_id: str) -> bool:
        """
        删除检查点数据
        
        Args:
            thread_id: 线程ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 使用delete_thread方法删除线程的所有检查点
            self.langchain_saver.delete_thread(thread_id)
            return True
        except Exception as e:
            print(f"删除检查点失败: {e}")
            return False

    def list_threads(self) -> List[str]:
        """
        列出所有线程ID
        
        Returns:
            List[str]: 线程ID列表
        """
        try:
            # 对于InMemorySaver，可以获取所有存储的线程
            if hasattr(self.langchain_saver, 'storage'):
                return list(self.langchain_saver.storage.keys())
            
            # 尝试使用list方法获取所有线程
            if hasattr(self.langchain_saver, 'list'):
                config = self._create_config("")
                threads = []
                for checkpoint_tuple in self.langchain_saver.list(config):
                    thread_id = checkpoint_tuple.config.get("configurable", {}).get("thread_id")
                    if thread_id and thread_id not in threads:
                        threads.append(thread_id)
                return threads
            
            return []
        except Exception as e:
            print(f"列出线程失败: {e}")
            return []

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    Checkpoint,
    CheckpointMetadata,
    ChannelVersions,
)
from langchain_core.runnables import RunnableConfig
from infrastructure.cache.redis_client import RedisClient
from collections.abc import Iterator, AsyncIterator
from typing import Any, Sequence
import pickle


class RedisCheckpointSaver(BaseCheckpointSaver):
    """Redis 检查点存储实现"""

    def __init__(self, redis_client: RedisClient | None = None):
        super().__init__()
        self.redis_client = redis_client or RedisClient().get_client()

    def _get_thread_key(self, thread_id: str) -> str:
        """获取线程存储的键"""
        return f"langgraph:thread:{thread_id}"

    def _get_checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        """获取检查点存储的键"""
        return f"langgraph:checkpoint:{thread_id}:{checkpoint_id}"

    def _get_writes_key(self, thread_id: str, checkpoint_id: str) -> str:
        """获取写入存储的键"""
        return f"langgraph:writes:{thread_id}:{checkpoint_id}"

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """获取检查点元组"""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str | None = config["configurable"].get("checkpoint_id")

        if checkpoint_id:
            # 根据检查点ID查找
            checkpoint_key = self._get_checkpoint_key(thread_id, checkpoint_id)
            checkpoint_data = self.redis_client.get(checkpoint_key)
            if not checkpoint_data:
                return None

            # 反序列化检查点
            checkpoint, metadata, parent_checkpoint_id = pickle.loads(checkpoint_data)

            # 获取待处理的写入
            writes_key = self._get_writes_key(thread_id, checkpoint_id)
            writes_data = self.redis_client.get(writes_key)
            pending_writes = pickle.loads(writes_data) if writes_data else None

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes=pending_writes,
            )
        else:
            # 获取最新的检查点
            thread_key = self._get_thread_key(thread_id)
            checkpoint_ids = self.redis_client.lrange(thread_key, 0, -1)
            if not checkpoint_ids:
                return None

            # 获取最新的检查点ID
            latest_checkpoint_id = checkpoint_ids[-1].decode("utf-8")
            checkpoint_key = self._get_checkpoint_key(thread_id, latest_checkpoint_id)
            checkpoint_data = self.redis_client.get(checkpoint_key)
            if not checkpoint_data:
                return None

            # 反序列化检查点
            checkpoint, metadata, parent_checkpoint_id = pickle.loads(checkpoint_data)

            # 获取待处理的写入
            writes_key = self._get_writes_key(thread_id, latest_checkpoint_id)
            writes_data = self.redis_client.get(writes_key)
            pending_writes = pickle.loads(writes_data) if writes_data else None

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": latest_checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes=pending_writes,
            )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """列出符合条件的检查点"""
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """保存检查点"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # 保存检查点
        checkpoint_key = self._get_checkpoint_key(thread_id, checkpoint_id)
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_data = pickle.dumps((checkpoint, metadata, parent_checkpoint_id))
        self.redis_client.set(checkpoint_key, checkpoint_data)

        # 保存到线程的检查点列表
        thread_key = self._get_thread_key(thread_id)
        self.redis_client.rpush(thread_key, checkpoint_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """保存待处理的写入"""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        writes_key = self._get_writes_key(thread_id, checkpoint_id)
        self.redis_client.set(writes_key, pickle.dumps(writes))

    def delete_thread(self, thread_id: str) -> None:
        """删除线程的所有检查点"""
        # 删除线程的检查点列表
        thread_key = self._get_thread_key(thread_id)
        checkpoint_ids = self.redis_client.lrange(thread_key, 0, -1)
        self.redis_client.delete(thread_key)

        # 删除所有检查点和写入
        for checkpoint_id in checkpoint_ids:
            checkpoint_id_str = checkpoint_id.decode("utf-8")
            self.redis_client.delete(
                self._get_checkpoint_key(thread_id, checkpoint_id_str)
            )
            self.redis_client.delete(
                self._get_writes_key(thread_id, checkpoint_id_str)
            )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """异步获取检查点元组"""
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """异步列出符合条件的检查点"""
        raise NotImplementedError
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """异步保存检查点"""
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """异步保存待处理的写入"""
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """异步删除线程的所有检查点"""
        return self.delete_thread(thread_id)


def create_redis_saver() -> RedisCheckpointSaver:
    """创建 Redis 检查点存储实例"""
    return RedisCheckpointSaver()

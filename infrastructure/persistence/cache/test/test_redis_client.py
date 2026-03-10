"""
Redis 客户端功能测试
"""

import pytest
from infrastructure.persistence.cache.redis_client import RedisClient
from infrastructure.persistence.cache.redis_saver import RedisCheckpointSaver, create_redis_saver
from infrastructure.persistence.cache.storage_factory import create_storage_saver
from domain.shared.enums import StorageBackend
from langgraph.checkpoint.memory import InMemorySaver


class TestRedisClient:
    """测试 Redis 客户端功能"""

    def test_singleton_instance(self):
        """测试单例模式是否正常工作"""
        client1 = RedisClient()
        client2 = RedisClient()
        assert client1 is client2
        assert client1.get_client() is client2.get_client()

    def test_client_creation(self):
        """测试 Redis 客户端实例创建"""
        client = RedisClient()
        assert client.get_client() is not None
        assert hasattr(client, "settings")

    @pytest.mark.skip(reason="需要运行 Redis 服务才能测试")
    def test_ping(self):
        """测试 Redis 连接（需要实际 Redis 服务）"""
        client = RedisClient()
        assert client.ping() is True


class TestRedisCheckpointSaver:
    """测试 Redis 检查点存储功能"""

    @pytest.mark.skip(reason="需要运行 Redis 服务才能测试")
    def test_saver_creation(self):
        """测试创建 RedisCheckpointSaver 实例"""
        saver = create_redis_saver()
        assert isinstance(saver, RedisCheckpointSaver)
        assert saver.redis_client is not None

    @pytest.mark.skip(reason="需要运行 Redis 服务才能测试")
    def test_save_and_retrieve_checkpoint(self):
        """测试保存和检索检查点"""
        saver = create_redis_saver()

        # 模拟一个简单的检查点
        config = {
            "configurable": {
                "thread_id": "test-thread-1",
                "checkpoint_ns": "test-ns",
            }
        }
        checkpoint = {
            "id": "test-checkpoint-1",
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"test-channel": "test-value"},
            "channel_versions": {"test-channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata = {"source": "test", "step": 0}
        new_versions = {"test-channel": "1"}

        # 保存检查点
        saved_config = saver.put(config, checkpoint, metadata, new_versions)

        # 检索检查点
        retrieved = saver.get_tuple(saved_config)

        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint["id"]
        assert retrieved.metadata == metadata

    @pytest.mark.skip(reason="需要运行 Redis 服务才能测试")
    def test_delete_thread(self):
        """测试删除线程的所有检查点"""
        saver = create_redis_saver()

        # 首先创建一个检查点
        config = {
            "configurable": {
                "thread_id": "test-thread-2",
                "checkpoint_ns": "test-ns",
            }
        }
        checkpoint = {
            "id": "test-checkpoint-2",
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"test-channel": "test-value"},
            "channel_versions": {"test-channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
            "updated_channels": None,
        }
        metadata = {"source": "test", "step": 0}
        new_versions = {"test-channel": "1"}

        saver.put(config, checkpoint, metadata, new_versions)

        # 检查检查点是否存在
        retrieved = saver.get_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread-2",
                    "checkpoint_ns": "test-ns",
                    "checkpoint_id": "test-checkpoint-2",
                }
            }
        )
        assert retrieved is not None

        # 删除线程
        saver.delete_thread("test-thread-2")

        # 验证检查点已删除
        retrieved_after_delete = saver.get_tuple(
            {
                "configurable": {
                    "thread_id": "test-thread-2",
                    "checkpoint_ns": "test-ns",
                    "checkpoint_id": "test-checkpoint-2",
                }
            }
        )
        assert retrieved_after_delete is None


class TestStorageFactory:
    """测试存储后端工厂"""

    def test_create_in_memory_saver(self):
        """测试创建 InMemorySaver"""
        saver = create_storage_saver(StorageBackend.IN_MEMORY.value)
        assert isinstance(saver, InMemorySaver)

    @pytest.mark.skip(reason="需要运行 Redis 服务才能测试")
    def test_create_redis_saver(self):
        """测试创建 RedisCheckpointSaver（需要实际 Redis 服务）"""
        saver = create_storage_saver(StorageBackend.REDIS.value)
        assert isinstance(saver, RedisCheckpointSaver)

    def test_invalid_backend(self):
        """测试无效的存储后端"""
        with pytest.raises(ValueError):
            create_storage_saver("invalid_backend")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

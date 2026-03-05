from langgraph.checkpoint.memory import InMemorySaver
from enums.enums import StorageBackend
from infrastructure.config.settings import settings
from infrastructure.cache.redis_saver import create_redis_saver


def create_storage_saver(backend: str | None = None) -> object:
    """创建存储后端实例"""
    backend = backend or settings.app.storage_backend

    if backend == StorageBackend.REDIS.value:
        return create_redis_saver()
    elif backend == StorageBackend.IN_MEMORY.value:
        return InMemorySaver()
    else:
        raise ValueError(f"不支持的存储后端类型: {backend}")

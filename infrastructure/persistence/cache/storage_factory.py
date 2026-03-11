"""
存储工厂模块 - 根据框架和后端创建存储适配器
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from domain.shared.enums import StorageBackend, Framework
from domain.shared.ports.checkpoint_port import CheckpointPort
from config.settings import settings
from infrastructure.persistence.cache.redis_saver import create_redis_saver
from infrastructure.persistence.cache.adapters.langchain_checkpoint_adapter import LangChainCheckpointAdapter


# ==================== 框架工厂 ====================

class CheckpointSaverFactory:
    """检查点存储工厂基类"""
    
    @staticmethod
    def create_saver(backend: str | None = None) -> BaseCheckpointSaver:
        """创建框架特定的存储后端"""
        raise NotImplementedError


class LangChainCheckpointSaverFactory(CheckpointSaverFactory):
    """LangChain检查点存储工厂"""
    
    @staticmethod
    def create_saver(backend: str | None = None) -> BaseCheckpointSaver:
        """创建LangChain存储后端"""
        backend = backend or settings.app.storage_backend
        
        if backend == StorageBackend.REDIS.value:
            return create_redis_saver()
        elif backend == StorageBackend.IN_MEMORY.value:
            return InMemorySaver()
        else:
            raise ValueError(f"不支持的存储后端类型: {backend}")


# 框架工厂注册表
_FRAMEWORK_FACTORIES: dict[str, type[CheckpointSaverFactory]] = {
    Framework.LANGCHAIN.value: LangChainCheckpointSaverFactory,
}


def get_framework_factory(framework: str | None = None) -> type[CheckpointSaverFactory]:
    """获取指定框架的工厂类"""
    framework = framework or settings.app.framework
    
    if framework not in _FRAMEWORK_FACTORIES:
        raise ValueError(f"不支持的框架类型: {framework}")
    
    return _FRAMEWORK_FACTORIES[framework]


# ==================== 存储适配器工厂 ====================

def create_storage_saver(backend: str | None = None, framework: str | None = None) -> BaseCheckpointSaver:
    """
    创建存储后端实例（保持向后兼容）
    
    Args:
        backend: 存储后端类型
        framework: 框架类型
        
    Returns:
        BaseCheckpointSaver: 存储后端实例
    """
    framework = framework or settings.app.framework
    factory = get_framework_factory(framework)
    return factory.create_saver(backend)


def create_storage_adapter(backend: str | None = None, framework: str | None = None) -> CheckpointPort:
    """
    创建统一的存储适配器实例
    
    Args:
        backend: 存储后端类型
        framework: 框架类型
        
    Returns:
        CheckpointPort: 统一的存储适配器接口
    """
    framework = framework or settings.app.framework
    
    # 根据框架创建存储后端
    saver = create_storage_saver(backend, framework)
    
    # 根据框架创建适配器
    if framework == Framework.LANGCHAIN.value:
        return LangChainCheckpointAdapter(saver)
    else:
        raise ValueError(f"不支持的框架类型: {framework}")

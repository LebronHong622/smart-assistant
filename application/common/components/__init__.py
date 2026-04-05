"""
组件适配器模块
自动注册所有组件适配器到注册表
"""
from application.common.component_registry import ComponentRegistry
from application.common.components.redis_component import RedisComponent
from application.common.components.milvus_component import MilvusComponent
from application.common.components.mysql_component import MySQLComponent


def auto_register_components() -> None:
    """
    自动注册所有组件到组件注册表
    该函数应该在应用启动时调用
    """
    registry = ComponentRegistry.get_instance()

    registry.register(RedisComponent, "redis")
    registry.register(MilvusComponent, "milvus")
    registry.register(MySQLComponent, "mysql")


# 导出组件类供外部使用
__all__ = [
    "RedisComponent",
    "MilvusComponent",
    "MySQLComponent",
    "auto_register_components",
]

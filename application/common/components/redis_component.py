"""
Redis 组件适配器
将 RedisClient 适配为统一的 Component 接口
"""
from infrastructure.persistence.cache.redis_client import RedisClient
from application.common.component import Component, ComponentStatus
from infrastructure.core.log import app_logger


class RedisComponent(Component):
    """
    Redis 组件适配器
    封装 RedisClient，提供统一的初始化和关闭接口
    """

    def __init__(self):
        super().__init__(component_name="redis")

    def initialize(self) -> None:
        """
        初始化 Redis 连接
        """
        app_logger.info("初始化 Redis 组件")
        self._set_status(ComponentStatus.INITIALIZING)

        try:
            redis_client = RedisClient()
            # 调用 ping 方法触发连接建立
            if not redis_client.ping():
                raise RuntimeError("Redis 连接测试失败")

            self._instance = redis_client
            self._set_status(ComponentStatus.RUNNING)
            app_logger.info("Redis 组件初始化成功")
        except Exception as e:
            self._set_status(ComponentStatus.FAILED)
            raise

    def shutdown(self) -> None:
        """
        关闭 Redis 连接
        """
        app_logger.info("关闭 Redis 组件")
        if self._instance and self._instance.client:
            self._instance.client.close()
        self._set_status(ComponentStatus.STOPPED)
        app_logger.info("Redis 组件已关闭")

    def health_check(self) -> bool:
        """
        Redis 健康检查
        :return: 是否健康
        """
        if self._instance:
            try:
                return self._instance.ping()
            except Exception as e:
                app_logger.error(f"Redis 健康检查失败: {e}")
                return False
        return False

    def get_instance(self) -> RedisClient:
        """
        获取 RedisClient 实例
        :return: RedisClient 实例
        """
        if self._instance is None:
            raise RuntimeError("Redis 组件未初始化")
        return self._instance

"""
Milvus 组件适配器
将 MilvusClient 适配为统一的 Component 接口
"""
from infrastructure.persistence.vector.milvus_client import MilvusClient
from application.common.component import Component, ComponentStatus
from infrastructure.core.log import app_logger


class MilvusComponent(Component):
    """
    Milvus 组件适配器
    封装 MilvusClient，提供统一的初始化和关闭接口
    """

    def __init__(self):
        super().__init__(component_name="milvus")

    def initialize(self) -> None:
        """
        初始化 Milvus 连接
        """
        app_logger.info("初始化 Milvus 组件")
        self._set_status(ComponentStatus.INITIALIZING)

        try:
            # MilvusClient 是单例，在 __init__ 时已自动连接
            milvus_client = MilvusClient()
            # 尝试获取集合列表来验证连接
            milvus_client.list_collections()

            self._instance = milvus_client
            self._set_status(ComponentStatus.RUNNING)
            app_logger.info("Milvus 组件初始化成功")
        except Exception as e:
            self._set_status(ComponentStatus.FAILED)
            raise

    def shutdown(self) -> None:
        """
        关闭 Milvus 连接
        """
        app_logger.info("关闭 Milvus 组件")
        if self._instance:
            self._instance.disconnect()
        self._set_status(ComponentStatus.STOPPED)
        app_logger.info("Milvus 组件已关闭")

    def health_check(self) -> bool:
        """
        Milvus 健康检查
        :return: 是否健康
        """
        if self._instance:
            try:
                # 尝试获取集合列表来验证连接
                self._instance.list_collections()
                return True
            except Exception as e:
                app_logger.error(f"Milvus 健康检查失败: {e}")
                return False
        return False

    def get_instance(self) -> MilvusClient:
        """
        获取 MilvusClient 实例
        :return: MilvusClient 实例
        """
        if self._instance is None:
            raise RuntimeError("Milvus 组件未初始化")
        return self._instance

"""
MySQL 组件适配器
将 MySQLClient 适配为统一的 Component 接口
"""
from infrastructure.persistence.database.mysql_client import MySQLClient
from application.common.component import Component, ComponentStatus
from infrastructure.core.log import app_logger


class MySQLComponent(Component):
    """
    MySQL 组件适配器
    封装 MySQLClient，提供统一的初始化和关闭接口
    """

    def __init__(self):
        super().__init__(component_name="mysql")

    def initialize(self) -> None:
        """
        初始化 MySQL 连接
        """
        app_logger.info("初始化 MySQL 组件")
        self._set_status(ComponentStatus.INITIALIZING)

        try:
            mysql_client = MySQLClient()
            # 调用 ping 方法进行连接测试
            if not mysql_client.ping():
                raise RuntimeError("MySQL 连接测试失败")

            self._instance = mysql_client
            self._set_status(ComponentStatus.RUNNING)
            app_logger.info("MySQL 组件初始化成功")
        except Exception as e:
            self._set_status(ComponentStatus.FAILED)
            raise

    def shutdown(self) -> None:
        """
        关闭 MySQL 连接
        """
        app_logger.info("关闭 MySQL 组件")
        if self._instance:
            self._instance.close()
        self._set_status(ComponentStatus.STOPPED)
        app_logger.info("MySQL 组件已关闭")

    def health_check(self) -> bool:
        """
        MySQL 健康检查
        :return: 是否健康
        """
        if self._instance:
            try:
                return self._instance.ping()
            except Exception as e:
                app_logger.error(f"MySQL 健康检查失败: {e}")
                return False
        return False

    def get_instance(self) -> MySQLClient:
        """
        获取 MySQLClient 实例
        :return: MySQLClient 实例
        """
        if self._instance is None:
            raise RuntimeError("MySQL 组件未初始化")
        return self._instance

"""
应用初始化器，负责统一管理所有底层组件的启动和关闭
遵循DDD架构，位于application层，不依赖interface层
"""
import signal
import sys
from typing import List, Dict, Any, Optional
from enum import Enum

from infrastructure.log import app_logger
from config.settings import get_app_settings
from infrastructure.cache.redis_client import RedisClient
from infrastructure.vector.milvus_client import MilvusClient
from infrastructure.database.postgres_client import PostgreSQLClient


class ComponentStatus(Enum):
    """组件状态枚举"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"


class AppInitializer:
    """
    应用初始化器，单例模式
    统一管理所有底层组件的生命周期
    """
    _instance: Optional["AppInitializer"] = None
    _initialized: bool = False

    def __new__(cls) -> "AppInitializer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._component_status: Dict[str, ComponentStatus] = {}
            cls._instance._component_instances: Dict[str, Any] = {}
            cls._instance._settings = get_app_settings()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "AppInitializer":
        """获取单例实例"""
        return cls()

    def initialize(self) -> None:
        """
        初始化所有配置的组件
        按照配置顺序依次初始化，失败时根据fail_fast配置决定是否终止
        """
        if self._initialized:
            app_logger.info("应用初始化器已初始化，跳过重复初始化")
            return

        app_logger.info("开始初始化应用底层组件")
        preload_components = self._settings.preload_components
        fail_fast = self._settings.fail_fast_on_init_error

        # 初始化组件状态
        for component in preload_components:
            self._component_status[component] = ComponentStatus.NOT_INITIALIZED

        # 按顺序初始化组件
        for component in preload_components:
            try:
                app_logger.info(f"正在初始化组件: {component}")
                self._component_status[component] = ComponentStatus.INITIALIZING

                if component == "redis":
                    self._init_redis()
                elif component == "milvus":
                    self._init_milvus()
                elif component == "postgres":
                    self._init_postgres()
                else:
                    app_logger.warning(f"未知组件类型: {component}，跳过初始化")
                    self._component_status[component] = ComponentStatus.STOPPED
                    continue

                self._component_status[component] = ComponentStatus.RUNNING
                app_logger.info(f"组件 {component} 初始化成功")

            except Exception as e:
                error_msg = f"组件 {component} 初始化失败: {str(e)}"
                app_logger.error(error_msg)
                self._component_status[component] = ComponentStatus.FAILED

                if fail_fast:
                    app_logger.critical("开启了fail_fast模式，初始化失败，程序终止")
                    raise RuntimeError(error_msg) from e

        self._initialized = True
        app_logger.info("应用底层组件初始化完成")
        # 注册信号处理函数
        self._register_signal_handlers()

    def _init_redis(self) -> None:
        """初始化Redis连接"""
        redis_client = RedisClient()
        # 调用ping方法触发连接建立
        if not redis_client.ping():
            raise RuntimeError("Redis连接测试失败")
        self._component_instances["redis"] = redis_client

    def _init_milvus(self) -> None:
        """初始化Milvus连接"""
        milvus_client = MilvusClient()
        # 调用connect方法建立连接
        milvus_client.connect()
        if not milvus_client.ping():
            raise RuntimeError("Milvus连接测试失败")
        self._component_instances["milvus"] = milvus_client

    def _init_postgres(self) -> None:
        """初始化PostgreSQL连接（待实现）"""
        # 待PostgreSQL客户端实现后完善
        # postgres_client = PostgreSQLClient()
        # if not postgres_client.ping():
        #     raise RuntimeError("PostgreSQL连接测试失败")
        # self._component_instances["postgres"] = postgres_client
        app_logger.warning("PostgreSQL组件尚未实现，跳过初始化")

    def shutdown(self) -> None:
        """
        统一关闭所有组件，释放资源
        按照初始化的逆序关闭
        """
        if not self._initialized:
            return

        app_logger.info("开始关闭应用底层组件")
        # 逆序关闭组件
        for component in reversed(list(self._component_status.keys())):
            status = self._component_status[component]
            if status != ComponentStatus.RUNNING:
                app_logger.info(f"组件 {component} 状态为 {status.value}，跳过关闭")
                continue

            try:
                app_logger.info(f"正在关闭组件: {component}")
                if component == "redis":
                    self._shutdown_redis()
                elif component == "milvus":
                    self._shutdown_milvus()
                elif component == "postgres":
                    self._shutdown_postgres()

                self._component_status[component] = ComponentStatus.STOPPED
                app_logger.info(f"组件 {component} 关闭成功")
            except Exception as e:
                app_logger.error(f"组件 {component} 关闭失败: {str(e)}")

        self._initialized = False
        app_logger.info("所有应用底层组件已关闭")

    def _shutdown_redis(self) -> None:
        """关闭Redis连接"""
        redis_client = self._component_instances.get("redis")
        if redis_client:
            redis_client.close()

    def _shutdown_milvus(self) -> None:
        """关闭Milvus连接"""
        milvus_client = self._component_instances.get("milvus")
        if milvus_client:
            milvus_client.close()

    def _shutdown_postgres(self) -> None:
        """关闭PostgreSQL连接"""
        postgres_client = self._component_instances.get("postgres")
        if postgres_client:
            postgres_client.close()

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查，返回所有组件的状态
        """
        health_status = {
            "app_status": "healthy" if all(
                status in (ComponentStatus.RUNNING, ComponentStatus.STOPPED)
                for status in self._component_status.values()
            ) else "unhealthy",
            "components": {}
        }

        for component, status in self._component_status.items():
            component_health = {
                "status": status.value,
                "is_healthy": status == ComponentStatus.RUNNING
            }

            # 对运行中的组件做实时健康检查
            if status == ComponentStatus.RUNNING:
                try:
                    if component == "redis":
                        component_health["is_healthy"] = self._component_instances["redis"].ping()
                    elif component == "milvus":
                        component_health["is_healthy"] = self._component_instances["milvus"].ping()
                    elif component == "postgres":
                        # 待实现
                        component_health["is_healthy"] = True
                except Exception as e:
                    component_health["is_healthy"] = False
                    component_health["error"] = str(e)

            if not component_health["is_healthy"] and component_health["status"] == "running":
                health_status["app_status"] = "unhealthy"

            health_status["components"][component] = component_health

        return health_status

    def _register_signal_handlers(self) -> None:
        """注册信号处理函数，实现优雅关闭"""
        def signal_handler(signal_num, frame):
            app_logger.info(f"收到信号 {signal_num}，开始优雅关闭程序")
            self.shutdown()
            sys.exit(0)

        # 注册常用信号
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windows系统支持SIGBREAK
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

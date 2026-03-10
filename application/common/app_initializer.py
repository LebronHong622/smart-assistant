"""
应用初始化器，负责统一管理所有底层组件的启动和关闭
遵循DDD架构，位于application层，不依赖interface层
使用适配器模式+策略模式+注册机制，实现对扩展开放，对修改关闭
"""
from typing import List, Dict, Any, Optional

from infrastructure.core.log import app_logger
from config.settings import get_app_settings
from application.common.component import Component, ComponentStatus
from application.common.component_registry import ComponentRegistry
from application.common.components import auto_register_components


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
            cls._instance._components: Dict[str, Component] = {}
            cls._instance._settings = get_app_settings()
            # 自动注册所有组件
            auto_register_components()
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
        preload_components = self._settings.app.preload_components
        fail_fast = self._settings.app.fail_fast_on_init_error

        # 获取组件注册表
        registry = ComponentRegistry.get_instance()

        # 按顺序初始化组件
        for component_name in preload_components:
            try:
                app_logger.info(f"正在初始化组件: {component_name}")

                # 从注册表获取组件类
                component_class = registry.get_component_class(component_name)
                if component_class is None:
                    app_logger.warning(f"未注册的组件类型: {component_name}，跳过初始化")
                    continue

                # 创建组件实例并初始化
                component_instance = component_class()
                self._components[component_name] = component_instance  # 先添加到字典，即使初始化失败也能获取
                component_instance.initialize()

                app_logger.info(f"组件 {component_name} 初始化成功")

            except Exception as e:
                error_msg = f"组件 {component_name} 初始化失败: {str(e)}"
                app_logger.error(error_msg)

                if fail_fast:
                    app_logger.critical("开启了fail_fast模式，初始化失败，程序终止")
                    raise RuntimeError(error_msg) from e
                # 不启用 fail_fast 时，组件已经在字典中，状态由组件内部设置为 FAILED

        self._initialized = True
        app_logger.info("应用底层组件初始化完成")

    def shutdown(self) -> None:
        """
        统一关闭所有组件，释放资源
        按照初始化的逆序关闭
        """
        if not self._initialized:
            return

        app_logger.info("开始关闭应用底层组件")
        # 逆序关闭组件
        for component_name in reversed(list(self._components.keys())):
            component = self._components[component_name]
            status = component.get_status()

            if status != ComponentStatus.RUNNING:
                app_logger.info(f"组件 {component_name} 状态为 {status.value}，跳过关闭")
                continue

            try:
                app_logger.info(f"正在关闭组件: {component_name}")
                component.shutdown()
                app_logger.info(f"组件 {component_name} 关闭成功")
            except Exception as e:
                app_logger.error(f"组件 {component_name} 关闭失败: {str(e)}")

        self._initialized = False
        app_logger.info("所有应用底层组件已关闭")

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查，返回所有组件的状态
        """
        health_status = {
            "app_status": "healthy",
            "components": {}
        }

        for component_name, component in self._components.items():
            status = component.get_status()
            component_health = {
                "status": status.value,
                "is_healthy": status == ComponentStatus.RUNNING
            }

            # 对运行中的组件做实时健康检查
            if status == ComponentStatus.RUNNING:
                try:
                    component_health["is_healthy"] = component.health_check()
                except Exception as e:
                    component_health["is_healthy"] = False
                    component_health["error"] = str(e)

            if not component_health["is_healthy"] and component_health["status"] == "running":
                health_status["app_status"] = "unhealthy"

            health_status["components"][component_name] = component_health

        return health_status

    def get_component(self, component_name: str) -> Optional[Component]:
        """
        获取组件实例
        :param component_name: 组件名称
        :return: 组件实例，如果不存在返回None
        """
        return self._components.get(component_name)

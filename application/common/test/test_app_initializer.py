"""
AppInitializer 单元测试
"""
import pytest
from unittest.mock import patch, MagicMock

from application.common.app_initializer import AppInitializer, ComponentStatus
from config.settings import get_app_settings


@pytest.fixture(autouse=True)
def reset_singleton():
    """重置单例实例，避免测试间干扰"""
    AppInitializer._instance = None
    yield
    AppInitializer._instance = None


def test_singleton_pattern():
    """测试单例模式"""
    instance1 = AppInitializer.get_instance()
    instance2 = AppInitializer.get_instance()
    assert instance1 is instance2


def test_initialize_success():
    """测试初始化成功场景"""
    settings = get_app_settings()
    # 只测试redis组件，避免依赖其他服务
    settings.preload_components = ["redis"]
    settings.fail_fast_on_init_error = True

    with patch('application.common.app_initializer.RedisClient') as mock_redis_class:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        initializer = AppInitializer.get_instance()
        initializer.initialize()

        assert initializer._component_status["redis"] == ComponentStatus.RUNNING
        assert initializer._component_instances["redis"] == mock_redis
        mock_redis.ping.assert_called_once()


def test_initialize_fail_fast():
    """测试初始化失败时fail_fast模式"""
    settings = get_app_settings()
    settings.preload_components = ["redis"]
    settings.fail_fast_on_init_error = True

    with patch('application.common.app_initializer.RedisClient') as mock_redis_class:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = False
        mock_redis_class.return_value = mock_redis

        initializer = AppInitializer.get_instance()

        with pytest.raises(RuntimeError, match="组件 redis 初始化失败"):
            initializer.initialize()

        assert initializer._component_status["redis"] == ComponentStatus.FAILED


def test_initialize_no_fail_fast():
    """测试初始化失败时不启用fail_fast模式"""
    settings = get_app_settings()
    settings.preload_components = ["redis"]
    settings.fail_fast_on_init_error = False

    with patch('application.common.app_initializer.RedisClient') as mock_redis_class:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = False
        mock_redis_class.return_value = mock_redis

        initializer = AppInitializer.get_instance()
        initializer.initialize()  # 不应该抛出异常

        assert initializer._component_status["redis"] == ComponentStatus.FAILED


def test_health_check():
    """测试健康检查功能"""
    settings = get_app_settings()
    settings.preload_components = ["redis", "milvus"]

    with patch('application.common.app_initializer.RedisClient') as mock_redis_class, \
         patch('application.common.app_initializer.MilvusClient') as mock_milvus_class:
        # Mock Redis
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        # Mock Milvus
        mock_milvus = MagicMock()
        mock_milvus.connect.return_value = None
        mock_milvus.ping.return_value = True
        mock_milvus_class.return_value = mock_milvus

        initializer = AppInitializer.get_instance()
        initializer.initialize()

        health_status = initializer.health_check()

        assert health_status["app_status"] == "healthy"
        assert health_status["components"]["redis"]["status"] == "running"
        assert health_status["components"]["redis"]["is_healthy"] is True
        assert health_status["components"]["milvus"]["status"] == "running"
        assert health_status["components"]["milvus"]["is_healthy"] is True

        # 测试异常场景
        mock_redis.ping.return_value = False
        health_status = initializer.health_check()
        assert health_status["app_status"] == "unhealthy"
        assert health_status["components"]["redis"]["is_healthy"] is False


def test_shutdown():
    """测试组件关闭功能"""
    settings = get_app_settings()
    settings.preload_components = ["redis", "milvus"]

    with patch('application.common.app_initializer.RedisClient') as mock_redis_class, \
         patch('application.common.app_initializer.MilvusClient') as mock_milvus_class:
        # Mock Redis
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.close.return_value = None
        mock_redis_class.return_value = mock_redis

        # Mock Milvus
        mock_milvus = MagicMock()
        mock_milvus.connect.return_value = None
        mock_milvus.ping.return_value = True
        mock_milvus.close.return_value = None
        mock_milvus_class.return_value = mock_milvus

        initializer = AppInitializer.get_instance()
        initializer.initialize()

        initializer.shutdown()

        mock_redis.close.assert_called_once()
        mock_milvus.close.assert_called_once()
        assert initializer._component_status["redis"] == ComponentStatus.STOPPED
        assert initializer._component_status["milvus"] == ComponentStatus.STOPPED


def test_unknown_component():
    """测试未知组件初始化"""
    settings = get_app_settings()
    settings.preload_components = ["unknown_component"]
    settings.fail_fast_on_init_error = True

    initializer = AppInitializer.get_instance()
    initializer.initialize()  # 不应该抛出异常

    assert initializer._component_status["unknown_component"] == ComponentStatus.STOPPED

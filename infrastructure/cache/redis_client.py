import redis
from config.settings import settings
from infrastructure.log import app_logger


class RedisClient:
    """Redis 客户端单例类"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.settings = settings.redis
            self.client = self._create_client()
            self._initialized = True

    def _create_client(self):
        """创建 Redis 客户端连接"""
        app_logger.info("正在创建 Redis 客户端连接")
        try:
            if self.settings.redis_url and self.settings.redis_url != "redis://localhost:6379/0":
                app_logger.debug(f"使用 URL 连接 Redis: {self.settings.redis_url}")
                client = redis.from_url(
                    self.settings.redis_url,
                    socket_timeout=self.settings.redis_socket_timeout,
                    socket_connect_timeout=self.settings.redis_socket_connect_timeout,
                    retry_on_timeout=self.settings.redis_retry_on_timeout,
                    max_connections=self.settings.redis_max_connections
                )
            else:
                app_logger.debug(f"使用参数连接 Redis: host={self.settings.redis_host}, port={self.settings.redis_port}, db={self.settings.redis_db}")
                client = redis.Redis(
                    host=self.settings.redis_host,
                    port=self.settings.redis_port,
                    db=self.settings.redis_db,
                    password=self.settings.redis_password,
                    socket_timeout=self.settings.redis_socket_timeout,
                    socket_connect_timeout=self.settings.redis_socket_connect_timeout,
                    retry_on_timeout=self.settings.redis_retry_on_timeout,
                    max_connections=self.settings.redis_max_connections
                )

            # 测试连接
            client.ping()
            app_logger.info("Redis 连接成功")
            return client

        except Exception as e:
            app_logger.error(f"Redis 连接失败: {str(e)}")
            raise ConnectionError(f"Redis 连接失败: {str(e)}") from e

    def get_client(self):
        """获取 Redis 客户端实例"""
        return self.client

    def ping(self):
        """测试 Redis 连接"""
        try:
            result = self.client.ping()
            app_logger.debug(f"Redis ping 结果: {result}")
            return result
        except Exception as e:
            app_logger.error(f"Redis ping 失败: {str(e)}")
            raise ConnectionError(f"Redis 连接失败: {str(e)}") from e
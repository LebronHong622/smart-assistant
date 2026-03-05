import redis
from infrastructure.config.settings import settings


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
        if self.settings.redis_url and self.settings.redis_url != "redis://localhost:6379/0":
            return redis.from_url(
                self.settings.redis_url,
                socket_timeout=self.settings.redis_socket_timeout,
                socket_connect_timeout=self.settings.redis_socket_connect_timeout,
                retry_on_timeout=self.settings.redis_retry_on_timeout,
                max_connections=self.settings.redis_max_connections
            )
        else:
            return redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                socket_timeout=self.settings.redis_socket_timeout,
                socket_connect_timeout=self.settings.redis_socket_connect_timeout,
                retry_on_timeout=self.settings.redis_retry_on_timeout,
                max_connections=self.settings.redis_max_connections
            )

    def get_client(self):
        """获取 Redis 客户端实例"""
        return self.client

    def ping(self):
        """测试 Redis 连接"""
        try:
            return self.client.ping()
        except Exception as e:
            raise ConnectionError(f"Redis 连接失败: {str(e)}") from e
"""
项目全局配置文件
使用pydantic库进行配置验证和类型检查
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from domain.shared.enums import OverflowMemoryMethod, StorageBackend, Framework


load_dotenv()

BASE_MODEL_CONFIG = {
    "env_prefix": "",
    "case_sensitive": False,
}

class APISettings(BaseSettings):
    # Deepseek API配置
    deepseek_api_key: str = Field(..., description="Deepseek API密钥")
    deepseek_api_base: str = Field("https://api.deepseek.com/v1", description="Deepseek API基础URL")
    model: str = Field("deepseek-chat", description="Deepseek模型名称")
    temperature: float = Field(0.7, description="Deepseek模型温度参数")
    max_tokens: int = Field(1024, description="Deepseek模型最大令牌数")

    # 高德地图API配置
    amap_api_key: str = Field(..., description="高德地图API密钥")
    amap_api_url: str = Field("https://restapi.amap.com/v3/weather/weatherInfo", description="高德地图天气查询API URL")
    amap_name_code_map_file_path: str = Field("infrastructure/tool/AMap_adcode_citycode.xlsx", description="高德地图城市名称到城市编码的映射文件路径")


    @field_validator("deepseek_api_key", "amap_api_key")
    @classmethod
    def validate_api_keys(cls, v: str) -> str:
        """验证API密钥不能为空"""
        if not v or v.strip() == "":
            raise ValueError("API密钥不能为空")
        return v.strip()

    model_config = BASE_MODEL_CONFIG

class RedisSettings(BaseSettings):
    """Redis 连接配置类"""
    redis_url: str = Field("redis://localhost:6379/0", description="Redis 连接 URL")
    redis_host: str = Field("localhost", description="Redis 主机地址")
    redis_port: int = Field(6379, description="Redis 端口")
    redis_db: int = Field(0, description="Redis 数据库索引")
    redis_password: str | None = Field(None, description="Redis 密码")
    redis_socket_timeout: int = Field(5, description="Redis 连接超时时间（秒）")
    redis_socket_connect_timeout: int = Field(5, description="Redis 连接建立超时时间（秒）")
    redis_retry_on_timeout: bool = Field(True, description="超时是否重试")
    redis_max_connections: int = Field(10, description="最大连接数")

    model_config = BASE_MODEL_CONFIG


class MilvusSettings(BaseSettings):
    """Milvus 向量数据库配置类"""
    milvus_host: str = Field("localhost", description="Milvus 主机地址")
    milvus_port: int = Field(19530, description="Milvus 端口")
    milvus_uri: str = Field("http://localhost:19530", description="Milvus 连接 URI")
    milvus_collection_name: str = Field("document_embeddings", description="Milvus 集合名称")
    milvus_dimension: int = Field(1536, description="向量维度")
    milvus_metric_type: str = Field("L2", description="相似度度量类型 (L2/IP)")
    milvus_index_type: str = Field("IVF_FLAT", description="索引类型")
    milvus_n_list: int = Field(1024, description="IVF 索引的 n_list 参数")
    milvus_n_probe: int = Field(10, description="")

    model_config = BASE_MODEL_CONFIG


class DashScopeSettings(BaseSettings):
    """DashScope API 配置类（阿里千文 Embeddings）"""
    dashscope_api_key: str = Field("", description="DashScope API 密钥")
    dashscope_embedding_model: str = Field("text-embedding-v3", description="DashScope Embeddings 模型名称")
    dashscope_embedding_dim: int = Field(768, description="DashScope Embeddings 向量维度 (text-embedding-v3 支持 1024/768/512)")

    @field_validator("dashscope_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """验证 API 密钥"""
        if not v or v.strip() == "" or v == "your_dashscope_api_key":
            raise ValueError("DashScope API 密钥不能为空，请在 .env 文件中配置")
        return v.strip()

    @field_validator("dashscope_embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """验证向量维度"""
        if v not in [1024, 768, 512]:
            raise ValueError("DashScope text-embedding-v3 模型只支持 1024, 768, 512 三种维度")
        return v

    model_config = BASE_MODEL_CONFIG


class AppSettings(BaseSettings):
    """
    应用全局配置类
    包含API配置和其他应用相关配置
    """
    log_level: str = Field("INFO", description="日志级别")
    max_session_history: int = Field(50, description="最大会话历史长度")
    max_tokens_before_summary: int = Field(4000, description="触发摘要的最大令牌数")
    overflow_memory_method: str = Field(OverflowMemoryMethod.SUMMARY.value, description="溢出内存管理方法")
    storage_backend: str = Field(StorageBackend.IN_MEMORY.value, description="会话存储后端 (in_memory/redis)")
    framework: str = Field(Framework.LANGCHAIN.value, description="AI框架类型 (langchain)")
    preload_components: list[str] = Field(["redis", "milvus", "postgres"], description="预加载的底层组件列表，逗号分隔")
    fail_fast_on_init_error: bool = Field(True, description="组件初始化失败时是否直接终止启动")
    use_langchain_components: bool = Field(False, description="是否使用LangChain兼容组件")
    langchain_embeddings_provider: str = Field("dashscope", description="LangChain Embeddings提供商")
    langchain_vector_store_provider: str = Field("milvus", description="LangChain VectorStore提供商")

    @field_validator("preload_components", mode="before")
    @classmethod
    def validate_preload_components(cls, v):
        """验证并转换预加载组件列表，支持逗号分隔字符串"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    model_config = BASE_MODEL_CONFIG


class DocumentStorageSettings(BaseSettings):
    """文档存储配置类"""
    document_storage_type: str = Field("local", description="文档存储类型 (local/milvus)")
    document_storage_path: str = Field("data/documents", description="本地文档存储路径")
    document_file_format: str = Field("json", description="文档文件格式 (json)")

    model_config = BASE_MODEL_CONFIG


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL 数据库配置类"""
    postgres_host: str = Field("localhost", description="PostgreSQL 主机地址")
    postgres_port: int = Field(5432, description="PostgreSQL 端口")
    postgres_user: str = Field("postgres", description="PostgreSQL 用户名")
    postgres_password: str = Field("postgres", description="PostgreSQL 密码")
    postgres_db: str = Field("smart-assistant", description="PostgreSQL 数据库名")
    postgres_url: str = Field(
        "postgresql://postgres:postgres@localhost:5432/smart-assistant",
        description="PostgreSQL 连接 URL"
    )

    model_config = BASE_MODEL_CONFIG

# 使用单例模式构建Settings实例
class Settings:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                self.api = APISettings()  # type: ignore
                self.app = AppSettings()  # type: ignore
                self.redis = RedisSettings()  # type: ignore
                self.milvus = MilvusSettings()  # type: ignore 新增：Milvus 配置属性
                self.dashscope = DashScopeSettings()  # type: ignore 新增：DashScope 配置属性
                self.document_storage = DocumentStorageSettings()  # type: ignore 新增：文档存储配置属性
                self.postgres = PostgreSQLSettings()  # type: ignore 新增：PostgreSQL 配置属性
                self._initialized = True
            except ValueError as e:
                raise RuntimeError(f"配置初始化失败: {str(e)}")

settings = Settings()


def get_app_settings() -> Settings:
    """
    获取全局配置单例
    :return: Settings 实例
    """
    return settings

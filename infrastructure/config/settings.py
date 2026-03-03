"""
项目全局配置文件
使用pydantic库进行配置验证和类型检查
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from enums.enums import OverflowMemoryMethod


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

class AppSettings(BaseSettings):
    """
    应用全局配置类
    包含API配置和其他应用相关配置
    """
    log_level: str = Field("INFO", description="日志级别")
    max_session_history: int = Field(50, description="最大会话历史长度")
    max_tokens_before_summary: int = Field(4000, description="触发摘要的最大令牌数")
    overflow_memory_method: str = Field(OverflowMemoryMethod.SUMMARY.value, description="溢出内存管理方法")

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
                self.api = APISettings() # type: ignore
                self.app = AppSettings() # type: ignore
                self._initialized = True
            except ValueError as e:
                raise RuntimeError(f"配置初始化失败: {str(e)}")

settings = Settings()

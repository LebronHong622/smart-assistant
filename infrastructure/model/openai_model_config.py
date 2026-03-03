"""
支持openai请求格式的模型配置
"""
from langchain_openai import ChatOpenAI
from infrastructure.config.settings import settings

MODEL_CONFIGS = {
    "deepseek-chat": ChatOpenAI(
        api_key=settings.api.deepseek_api_key,
        base_url=settings.api.deepseek_api_base,
        model="deepseek-chat",
        temperature=settings.api.temperature,
        max_tokens=settings.api.max_tokens
    )
}

DEFAULT_MODEL_NAME = "deepseek-chat"

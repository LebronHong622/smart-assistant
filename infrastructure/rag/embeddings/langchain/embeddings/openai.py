"""
OpenAI 兼容嵌入实现
支持 DeepSeek 等兼容 OpenAI 格式的 API
"""
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from config.settings import settings


def create_openai_embedding(
    model: str = "text-embedding-ada-002",
) -> Embeddings:
    """创建 OpenAI 嵌入函数"""
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=settings.api.deepseek_api_key,
        openai_api_base=settings.api.deepseek_api_base,
    )

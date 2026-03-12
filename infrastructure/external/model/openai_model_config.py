"""
支持openai请求格式的模型配置
"""
from .llm.adapters.langchain_chat_adapter import DeepSeekChatModel
from config.settings import settings

MODEL_CONFIGS = {
    "deepseek-chat": DeepSeekChatModel()
}

DEFAULT_MODEL_NAME = "deepseek-chat"

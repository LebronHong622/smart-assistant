"""
统一模型工厂入口
整合 Embedding 和 LLM 工厂
"""

from typing import Any, Optional
from infrastructure.external.model.embedding.factories.embedding_factory import EmbeddingFactory
from infrastructure.external.model.llm.factories.llm_factory import LLMFactory


class ModelFactory:
    """统一模型工厂入口"""

    @staticmethod
    def get_embedding(name: str = "dashscope"):
        """获取嵌入向量适配器"""
        return EmbeddingFactory.get(name)

    @staticmethod
    def get_llm(adapter_name: str = "default", model_name: Optional[str] = None) -> Any:
        """获取LLM模型"""
        return LLMFactory.get_model(adapter_name, model_name)

    @staticmethod
    def list_embedding_adapters() -> list[str]:
        """列出所有嵌入向量适配器"""
        return EmbeddingFactory.list_adapters()

    @staticmethod
    def list_llm_adapters() -> list[str]:
        """列出所有LLM适配器"""
        return LLMFactory.list_adapters()

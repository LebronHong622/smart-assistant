"""
LangChain 嵌入工厂
根据 provider 分发调用不同的嵌入实现
"""
from typing import Any, Dict, List, Optional, Type, Callable
from langchain_core.embeddings import Embeddings
from config.rag_settings import rag_settings
from domain.shared.ports import EmbeddingFactoryPort
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from infrastructure.core.log import app_logger
from infrastructure.rag.embeddings.langchain.embedding_adapter import LangChainEmbeddingAdapter


class LangChainEmbeddingFactory(EmbeddingFactoryPort):
    """
    LangChain 嵌入函数工厂

    创建 LangChain Embeddings 并包装为领域 EmbeddingGeneratorPort 接口。
    """

    _registered_embeddings: Dict[str, Type[Embeddings]] = {}

    @classmethod
    def register_embedding(cls, name: str, embedding_class: Type[Embeddings]) -> None:
        """注册自定义嵌入函数"""
        cls._registered_embeddings[name] = embedding_class
        app_logger.info(f"注册嵌入函数: {name}")

    @classmethod
    def create_embedding(
        cls,
        provider: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingGeneratorPort:
        """创建嵌入函数实例，返回领域接口"""
        provider = provider or rag_settings.rag_pipeline.default_embedding_model
        app_logger.info(f"创建嵌入函数: provider={provider}")

        # 获取嵌入维度
        dimension = cls.get_embedding_dimension(provider)

        # 创建 LangChain Embeddings（延迟导入各个 provider）
        if provider in cls._registered_embeddings:
            embedding_class = cls._registered_embeddings[provider]
            lc_embeddings = embedding_class(**kwargs)
        else:
            creators: Dict[str, Callable] = {}
            if provider == "dashscope":
                from infrastructure.rag.embeddings.langchain.embeddings.dashscope import create_dashscope_embedding
                creators["dashscope"] = create_dashscope_embedding
            elif provider == "openai":
                from infrastructure.rag.embeddings.langchain.embeddings.openai import create_openai_embedding
                creators["openai"] = create_openai_embedding
            elif provider == "huggingface":
                from infrastructure.rag.embeddings.langchain.embeddings.huggingface import create_huggingface_embedding
                creators["huggingface"] = create_huggingface_embedding

            if provider not in creators:
                raise ValueError(f"不支持的嵌入函数类型: {provider}")

            lc_embeddings = creators[provider](**kwargs)

        # 包装为领域接口
        return LangChainEmbeddingAdapter(lc_embeddings, dimension=dimension)

    @classmethod
    def get_embedding_dimension(cls, provider: Optional[str] = None) -> int:
        """获取嵌入向量的维度"""
        provider = provider or rag_settings.rag_pipeline.default_embedding_model

        config = rag_settings.get_embedding_config(provider)
        if config:
            return config.dimension

        default_dimensions = {
            "dashscope": 768,
            "openai": 1536,
            "huggingface": 384,
        }

        return default_dimensions.get(provider, 768)

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """列出所有支持的嵌入函数类型"""
        providers = {"dashscope", "openai", "huggingface"}
        providers.update(cls._registered_embeddings.keys())
        providers.update(rag_settings.embedding_models.keys())
        return list(providers)

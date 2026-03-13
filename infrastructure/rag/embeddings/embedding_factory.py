"""
嵌入函数工厂
根据配置动态创建嵌入函数实例
"""

from typing import Any, Dict, List, Optional, Type

from langchain_core.embeddings import Embeddings

from config.rag_settings import rag_settings, EmbeddingModelConfig
from config.settings import settings
from infrastructure.core.log import app_logger


class EmbeddingFactory:
    """
    嵌入函数工厂类
    
    根据配置动态创建不同类型的嵌入函数
    支持 DashScope、OpenAI 等嵌入模型
    """

    # 已注册的嵌入函数类
    _registered_embeddings: Dict[str, Type[Embeddings]] = {}

    @classmethod
    def register_embedding(cls, name: str, embedding_class: Type[Embeddings]) -> None:
        """
        注册自定义嵌入函数
        
        Args:
            name: 嵌入函数名称
            embedding_class: 嵌入函数类
        """
        cls._registered_embeddings[name] = embedding_class
        app_logger.info(f"注册嵌入函数: {name}")

    @classmethod
    def create_dashscope_embedding(
        cls,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Embeddings:
        """
        创建 DashScope 嵌入函数
        
        Args:
            model: 模型名称
            dimension: 向量维度
            batch_size: 批处理大小
            
        Returns:
            DashScope 嵌入函数实例
        """
        from langchain_community.embeddings import DashScopeEmbeddings
        
        config = rag_settings.get_embedding_config("dashscope")
        
        return DashScopeEmbeddings(
            model=model or (config.model if config else "text-embedding-v3"),
            dashscope_api_key=settings.dashscope.dashscope_api_key,
            batch_size=batch_size or (config.batch_size if config else 25),
        )

    @classmethod
    def create_openai_embedding(
        cls,
        model: str = "text-embedding-ada-002",
    ) -> Embeddings:
        """
        创建 OpenAI 嵌入函数
        
        Args:
            model: 模型名称
            
        Returns:
            OpenAI 嵌入函数实例
        """
        from langchain_openai import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=settings.api.deepseek_api_key,
            openai_api_base=settings.api.deepseek_api_base,
        )

    @classmethod
    def create_huggingface_embedding(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Embeddings:
        """
        创建 HuggingFace 嵌入函数
        
        Args:
            model_name: 模型名称
            
        Returns:
            HuggingFace 嵌入函数实例
        """
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        return HuggingFaceEmbeddings(model_name=model_name)

    @classmethod
    def create_embedding(
        cls,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Embeddings:
        """
        创建嵌入函数实例
        
        Args:
            provider: 嵌入函数提供者名称，默认从配置读取
            **kwargs: 额外参数
            
        Returns:
            嵌入函数实例
            
        Raises:
            ValueError: 不支持的嵌入函数类型
        """
        provider = provider or rag_settings.rag_pipeline.default_embedding_model
        app_logger.info(f"创建嵌入函数: provider={provider}")

        # 检查是否已注册
        if provider in cls._registered_embeddings:
            embedding_class = cls._registered_embeddings[provider]
            return embedding_class(**kwargs)

        # 内置支持
        creators = {
            "dashscope": cls.create_dashscope_embedding,
            "openai": cls.create_openai_embedding,
            "huggingface": cls.create_huggingface_embedding,
        }

        if provider not in creators:
            raise ValueError(f"不支持的嵌入函数类型: {provider}")

        return creators[provider](**kwargs)

    @classmethod
    def get_embedding_dimension(cls, provider: Optional[str] = None) -> int:
        """
        获取嵌入向量的维度
        
        Args:
            provider: 嵌入函数提供者
            
        Returns:
            向量维度
        """
        provider = provider or rag_settings.rag_pipeline.default_embedding_model
        
        config = rag_settings.get_embedding_config(provider)
        if config:
            return config.dimension
        
        # 默认维度
        default_dimensions = {
            "dashscope": 1536,
            "openai": 1536,
            "huggingface": 384,
        }
        
        return default_dimensions.get(provider, 768)

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """
        列出所有支持的嵌入函数类型
        
        Returns:
            嵌入函数类型列表
        """
        providers = {"dashscope", "openai", "huggingface"}
        providers.update(cls._registered_embeddings.keys())
        providers.update(rag_settings.embedding_models.keys())
        return list(providers)

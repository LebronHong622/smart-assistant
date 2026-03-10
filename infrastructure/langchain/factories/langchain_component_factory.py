"""
LangChain 组件工厂
统一创建和管理 LangChain 兼容组件，对外提供简单接口
"""
from typing import Dict, Type, Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from config.settings import get_app_settings
from infrastructure.external.model.embedding.adapters.langchain_embeddings_adapter import LangChainEmbeddingsAdapter
from infrastructure.persistence.vector.adapters.langchain_milvus_adapter import LangChainMilvusAdapter

settings = get_app_settings()

class LangChainComponentFactory:
    """
    LangChain 组件工厂
    单例模式管理组件实例，支持注册不同提供商的实现
    """
    _instance: Optional["LangChainComponentFactory"] = None
    _embeddings_providers: Dict[str, Type[Embeddings]] = {}
    _vector_store_providers: Dict[str, Type[VectorStore]] = {}
    _embeddings_instances: Dict[str, Embeddings] = {}
    _vector_store_instances: Dict[str, VectorStore] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 注册默认组件
            cls._instance._register_default_components()
        return cls._instance

    def _register_default_components(self):
        """注册默认的组件提供商"""
        # 注册 Embeddings 提供商
        self.register_embeddings_provider("dashscope", LangChainEmbeddingsAdapter)
        # 注册 VectorStore 提供商
        self.register_vector_store_provider("milvus", LangChainMilvusAdapter)

    def register_embeddings_provider(self, name: str, provider_class: Type[Embeddings]):
        """
        注册 Embeddings 提供商
        :param name: 提供商名称
        :param provider_class: 提供商类
        """
        self._embeddings_providers[name] = provider_class

    def register_vector_store_provider(self, name: str, provider_class: Type[VectorStore]):
        """
        注册 VectorStore 提供商
        :param name: 提供商名称
        :param provider_class: 提供商类
        """
        self._vector_store_providers[name] = provider_class

    def get_embeddings(self, provider: Optional[str] = None) -> Embeddings:
        """
        获取 Embeddings 实例
        :param provider: 提供商名称，默认使用配置中的 langchain_embeddings_provider
        :return: Embeddings 实例
        """
        provider = provider or settings.app.langchain_embeddings_provider
        if provider not in self._embeddings_instances:
            if provider not in self._embeddings_providers:
                raise ValueError(f"未注册的 Embeddings 提供商: {provider}")
            self._embeddings_instances[provider] = self._embeddings_providers[provider]()
        return self._embeddings_instances[provider]

    def get_vector_store(
        self,
        provider: Optional[str] = None,
        embeddings: Optional[Embeddings] = None
    ) -> VectorStore:
        """
        获取 VectorStore 实例
        :param provider: 提供商名称，默认使用配置中的 langchain_vector_store_provider
        :param embeddings: 自定义 Embeddings 实例，默认使用配置的 Embeddings
        :return: VectorStore 实例
        """
        provider = provider or settings.app.langchain_vector_store_provider
        cache_key = f"{provider}_{id(embeddings)}" if embeddings else provider

        if cache_key not in self._vector_store_instances:
            if provider not in self._vector_store_providers:
                raise ValueError(f"未注册的 VectorStore 提供商: {provider}")
            if embeddings is None:
                embeddings = self.get_embeddings()
            self._vector_store_instances[cache_key] = self._vector_store_providers[provider](
                embeddings=embeddings
            )
        return self._vector_store_instances[cache_key]

    def list_embeddings_providers(self) -> list[str]:
        """列出所有已注册的 Embeddings 提供商"""
        return list(self._embeddings_providers.keys())

    def list_vector_store_providers(self) -> list[str]:
        """列出所有已注册的 VectorStore 提供商"""
        return list(self._vector_store_providers.keys())

# 全局工厂实例
langchain_component_factory = LangChainComponentFactory()

# 便捷导出函数
def get_langchain_embeddings(provider: Optional[str] = None) -> Embeddings:
    """便捷获取 LangChain Embeddings 实例"""
    return langchain_component_factory.get_embeddings(provider)

def get_langchain_vector_store(
    provider: Optional[str] = None,
    embeddings: Optional[Embeddings] = None
) -> VectorStore:
    """便捷获取 LangChain VectorStore 实例"""
    return langchain_component_factory.get_vector_store(provider, embeddings)

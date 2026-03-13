"""
嵌入函数工厂
根据配置动态创建嵌入函数实例

已重构为 RAGComponentFactory 的代理，保持向后兼容
"""

from typing import Any, List, Optional

from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory


class EmbeddingFactory:
    """
    嵌入函数工厂类
    
    根据配置动态创建不同类型的嵌入函数
    支持 DashScope、OpenAI 等嵌入模型
    
    注意: 此类现在是 RAGComponentFactory 的代理
    实际实现根据 settings.app.framework 配置动态决定
    """

    @classmethod
    def register_embedding(cls, name: str, embedding_class: Any) -> None:
        """注册自定义嵌入函数"""
        factory = RAGComponentFactory.get_embedding_factory()
        if hasattr(factory, 'register_embedding'):
            factory.register_embedding(name, embedding_class)

    @classmethod
    def create_dashscope_embedding(
        cls,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Any:
        """创建 DashScope 嵌入函数"""
        factory = RAGComponentFactory.get_embedding_factory()
        return factory.create_dashscope_embedding(model, dimension, batch_size)

    @classmethod
    def create_openai_embedding(
        cls,
        model: str = "text-embedding-ada-002",
    ) -> Any:
        """创建 OpenAI 嵌入函数"""
        factory = RAGComponentFactory.get_embedding_factory()
        return factory.create_openai_embedding(model)

    @classmethod
    def create_huggingface_embedding(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Any:
        """创建 HuggingFace 嵌入函数"""
        factory = RAGComponentFactory.get_embedding_factory()
        return factory.create_huggingface_embedding(model_name)

    @classmethod
    def create_embedding(
        cls,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """创建嵌入函数实例"""
        factory = RAGComponentFactory.get_embedding_factory()
        return factory.create_embedding(provider, **kwargs)

    @classmethod
    def get_embedding_dimension(cls, provider: Optional[str] = None) -> int:
        """获取嵌入向量的维度"""
        factory = RAGComponentFactory.get_embedding_factory()
        return factory.get_embedding_dimension(provider)

    @classmethod
    def list_supported_providers(cls) -> List[str]:
        """列出所有支持的嵌入函数类型"""
        factory = RAGComponentFactory.get_embedding_factory()
        return factory.list_supported_providers()

"""
文本分块器工厂
根据配置动态创建文本分块器实例

已重构为 RAGComponentFactory 的代理，保持向后兼容
"""

from typing import Any, List, Optional

from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory


class TextSplitterFactory:
    """
    文本分块器工厂类
    
    根据配置动态创建不同类型的文本分块器
    支持扩展新的分块器类型
    
    注意: 此类现在是 RAGComponentFactory 的代理
    实际实现根据 settings.app.framework 配置动态决定
    """

    @classmethod
    def register_splitter(cls, name: str, splitter_class: Any) -> None:
        """注册自定义分块器"""
        factory = RAGComponentFactory.get_splitter_factory()
        if hasattr(factory, 'register_splitter'):
            factory.register_splitter(name, splitter_class)

    @classmethod
    def get_splitter_class(cls, splitter_type: str) -> Optional[Any]:
        """获取分块器类"""
        factory = RAGComponentFactory.get_splitter_factory()
        return factory.get_splitter_class(splitter_type)

    @classmethod
    def create_splitter(
        cls,
        splitter_type: str = "recursive",
        **kwargs
    ) -> Any:
        """创建文本分块器实例"""
        factory = RAGComponentFactory.get_splitter_factory()
        return factory.create_splitter(splitter_type, **kwargs)

    @classmethod
    def split_documents(
        cls,
        documents: List[Any],
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[Any]:
        """分块文档"""
        factory = RAGComponentFactory.get_splitter_factory()
        return factory.split_documents(documents, splitter_type, **kwargs)

    @classmethod
    def split_text(
        cls,
        text: str,
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[str]:
        """分块文本"""
        factory = RAGComponentFactory.get_splitter_factory()
        return factory.split_text(text, splitter_type, **kwargs)

    @classmethod
    def create_code_splitter(
        cls,
        language: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Any:
        """创建代码分块器"""
        factory = RAGComponentFactory.get_splitter_factory()
        return factory.create_code_splitter(language, chunk_size, chunk_overlap)

    @classmethod
    def list_supported_splitters(cls) -> List[str]:
        """列出所有支持的分块器类型"""
        factory = RAGComponentFactory.get_splitter_factory()
        return factory.list_supported_splitters()

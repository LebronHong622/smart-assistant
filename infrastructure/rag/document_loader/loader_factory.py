"""
文档加载器工厂
根据配置动态创建文档加载器实例

已重构为 RAGComponentFactory 的代理，保持向后兼容
"""

from typing import Any, List, Optional

from domain.entity.document.document import Document
from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory


class DocumentLoaderFactory:
    """
    文档加载器工厂类
    
    根据配置动态创建不同类型的文档加载器
    支持扩展新的加载器类型
    
    注意: 此类现在是 RAGComponentFactory 的代理
    实际实现根据 settings.app.framework 配置动态决定
    """

    @classmethod
    def register_loader(cls, name: str, loader_class: Any) -> None:
        """注册自定义加载器"""
        factory = RAGComponentFactory.get_loader_factory()
        if hasattr(factory, 'register_loader'):
            factory.register_loader(name, loader_class)

    @classmethod
    def get_loader_class(cls, loader_type: str) -> Optional[Any]:
        """获取加载器类"""
        factory = RAGComponentFactory.get_loader_factory()
        return factory.get_loader_class(loader_type)

    @classmethod
    def create_loader(
        cls,
        loader_type: str,
        file_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建文档加载器实例"""
        factory = RAGComponentFactory.get_loader_factory()
        return factory.create_loader(loader_type, file_path, **kwargs)

    @classmethod
    def load_documents(
        cls,
        loader_type: str,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """加载文档，返回领域 Document 列表"""
        factory = RAGComponentFactory.get_loader_factory()
        return factory.load_documents(loader_type, file_path, **kwargs)

    @classmethod
    def load_from_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: str = "pdf",
        **kwargs
    ) -> List[Document]:
        """从目录加载文档，返回领域 Document 列表"""
        factory = RAGComponentFactory.get_loader_factory()
        return factory.load_from_directory(directory_path, glob_pattern, loader_type, **kwargs)

    @classmethod
    async def aload_documents(
        cls,
        loader_type: str,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """异步加载文档，返回领域 Document 列表"""
        factory = RAGComponentFactory.get_loader_factory()
        return await factory.aload_documents(loader_type, file_path, **kwargs)

    @classmethod
    async def aload_from_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: str = "pdf",
        **kwargs
    ) -> List[Document]:
        """异步从目录加载文档，返回领域 Document 列表"""
        factory = RAGComponentFactory.get_loader_factory()
        return await factory.aload_from_directory(directory_path, glob_pattern, loader_type, **kwargs)

    @classmethod
    def list_supported_loaders(cls) -> List[str]:
        """列出所有支持的加载器类型"""
        factory = RAGComponentFactory.get_loader_factory()
        return factory.list_supported_loaders()

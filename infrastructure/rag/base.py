"""
RAG 组件抽象接口
定义统一的组件接口，支持多框架实现
"""

from typing import Any, Dict, List, Optional, Type, Protocol, runtime_checkable


@runtime_checkable
class ILoaderFactory(Protocol):
    """文档加载器工厂接口"""

    def get_loader_class(self, loader_type: str) -> Optional[Type[Any]]:
        """获取加载器类"""
        ...

    def create_loader(
        self,
        loader_type: str,
        file_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建文档加载器实例"""
        ...

    def load_documents(
        self,
        loader_type: str,
        file_path: str,
        **kwargs
    ) -> List[Any]:
        """加载文档"""
        ...

    def list_supported_loaders(self) -> List[str]:
        """列出所有支持的加载器类型"""
        ...


@runtime_checkable
class IEmbeddingFactory(Protocol):
    """嵌入函数工厂接口"""

    def create_embedding(
        self,
        provider: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建嵌入函数实例"""
        ...

    def get_embedding_dimension(self, provider: Optional[str] = None) -> int:
        """获取嵌入向量的维度"""
        ...

    def list_supported_providers(self) -> List[str]:
        """列出所有支持的嵌入函数类型"""
        ...


@runtime_checkable
class IVectorStoreFactory(Protocol):
    """向量存储工厂接口"""

    def create_store(
        self,
        embedding: Any,
        collection_name: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建向量存储实例"""
        ...

    def get_store_config(self, provider: Optional[str] = None) -> Any:
        """获取向量存储配置"""
        ...

    def list_supported_providers(self) -> List[str]:
        """列出所有支持的向量存储类型"""
        ...


@runtime_checkable
class ISplitterFactory(Protocol):
    """文本分割器工厂接口"""

    def create_splitter(
        self,
        splitter_type: str = "recursive",
        **kwargs
    ) -> Any:
        """创建文本分割器实例"""
        ...

    def split_documents(
        self,
        documents: List[Any],
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[Any]:
        """分割文档"""
        ...

    def split_text(
        self,
        text: str,
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[str]:
        """分割文本"""
        ...

    def list_supported_splitters(self) -> List[str]:
        """列出所有支持的分割器类型"""
        ...

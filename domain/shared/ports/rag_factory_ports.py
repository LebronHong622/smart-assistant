"""
RAG 工厂端口定义
定义文档加载、嵌入、向量存储、文本分割等工厂的抽象接口
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from domain.document.entity.document import Document
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort


class LoaderFactoryPort(ABC):
    """文档加载器工厂端口"""

    @abstractmethod
    def get_loader_class(self, loader_type: str) -> Optional[Type[Any]]:
        """获取加载器类"""
        ...

    @abstractmethod
    def create_loader(
        self,
        loader_type: str,
        file_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建文档加载器实例"""
        ...

    @abstractmethod
    def load_documents(
        self,
        loader_type: str,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """加载文档，返回领域 Document 列表"""
        ...

    @abstractmethod
    def list_supported_loaders(self) -> List[str]:
        """列出所有支持的加载器类型"""
        ...


class EmbeddingFactoryPort(ABC):
    """嵌入函数工厂端口"""

    @abstractmethod
    def create_embedding(
        self,
        provider: Optional[str] = None,
        **kwargs
    ) -> EmbeddingGeneratorPort:
        """创建嵌入函数实例，返回领域接口"""
        ...

    @abstractmethod
    def get_embedding_dimension(self, provider: Optional[str] = None) -> int:
        """获取嵌入向量的维度"""
        ...

    @abstractmethod
    def list_supported_providers(self) -> List[str]:
        """列出所有支持的嵌入函数类型"""
        ...


class VectorStoreFactoryPort(ABC):
    """向量存储工厂端口"""

    @abstractmethod
    def create_store(
        self,
        embedding: Any,
        collection_name: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> Any:
        """创建向量存储实例"""
        ...

    @abstractmethod
    def get_store_config(self, provider: Optional[str] = None) -> Any:
        """获取向量存储配置"""
        ...

    @abstractmethod
    def list_supported_providers(self) -> List[str]:
        """列出所有支持的向量存储类型"""
        ...


class SplitterFactoryPort(ABC):
    """文本分割器工厂端口"""

    @abstractmethod
    def create_splitter(
        self,
        splitter_type: str = "recursive",
        **kwargs
    ) -> Any:
        """创建文本分割器实例"""
        ...

    @abstractmethod
    def split_documents(
        self,
        documents: List[Document],
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[Document]:
        """分割文档，接受并返回领域 Document 列表"""
        ...

    @abstractmethod
    def split_text(
        self,
        text: str,
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[str]:
        """分割文本"""
        ...

    @abstractmethod
    def list_supported_splitters(self) -> List[str]:
        """列出所有支持的分割器类型"""
        ...

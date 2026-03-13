"""
文档加载器工厂
根据配置动态创建文档加载器实例
"""

from typing import Any, Dict, List, Optional, Type
from importlib import import_module

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    DirectoryLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
    UnstructuredImageLoader,
)
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

from config.rag_settings import rag_settings, LoaderItemConfig
from infrastructure.core.log import app_logger


class DocumentLoaderFactory:
    """
    文档加载器工厂类
    
    根据配置动态创建不同类型的文档加载器
    支持扩展新的加载器类型
    """

    # 内置加载器映射
    _builtin_loaders: Dict[str, Type[BaseLoader]] = {
        "pdf": PyPDFLoader,
        "text": TextLoader,
        "csv": CSVLoader,
        "json": JSONLoader,
        "directory": DirectoryLoader,
        "web": WebBaseLoader,
        "markdown": UnstructuredMarkdownLoader,
        "image": UnstructuredImageLoader,
    }

    # 自定义加载器注册表
    _custom_loaders: Dict[str, Type[BaseLoader]] = {}

    @classmethod
    def register_loader(cls, name: str, loader_class: Type[BaseLoader]) -> None:
        """
        注册自定义加载器
        
        Args:
            name: 加载器名称
            loader_class: 加载器类
        """
        cls._custom_loaders[name] = loader_class
        app_logger.info(f"注册自定义文档加载器: {name}")

    @classmethod
    def get_loader_class(cls, loader_type: str) -> Optional[Type[BaseLoader]]:
        """
        获取加载器类
        
        Args:
            loader_type: 加载器类型
            
        Returns:
            加载器类或 None
        """
        # 先查找自定义加载器
        if loader_type in cls._custom_loaders:
            return cls._custom_loaders[loader_type]
        
        # 再查找内置加载器
        if loader_type in cls._builtin_loaders:
            return cls._builtin_loaders[loader_type]
        
        # 最后尝试从配置动态加载
        config = rag_settings.get_loader_config(loader_type)
        if config and config.enabled:
            try:
                module = import_module(config.module)
                loader_class = getattr(module, config.class_name)
                cls._custom_loaders[loader_type] = loader_class
                return loader_class
            except (ImportError, AttributeError) as e:
                app_logger.error(f"加载文档加载器失败 [{loader_type}]: {e}")
        
        return None

    @classmethod
    def create_loader(
        cls,
        loader_type: str,
        file_path: Optional[str] = None,
        **kwargs
    ) -> BaseLoader:
        """
        创建文档加载器实例
        
        Args:
            loader_type: 加载器类型
            file_path: 文件路径（某些加载器需要）
            **kwargs: 额外参数
            
        Returns:
            加载器实例
            
        Raises:
            ValueError: 不支持的加载器类型
        """
        # 获取配置
        config = rag_settings.get_loader_config(loader_type)
        if not config or not config.enabled:
            raise ValueError(f"不支持的文档加载器类型: {loader_type}")

        # 合并配置参数
        loader_kwargs = {**config.config, **kwargs}
        
        # 获取加载器类
        loader_class = cls.get_loader_class(loader_type)
        if not loader_class:
            raise ValueError(f"找不到文档加载器类: {loader_type}")

        app_logger.info(f"创建文档加载器: {loader_type}, 文件: {file_path}")

        # 根据加载器类型创建实例
        if file_path:
            return loader_class(file_path, **loader_kwargs)
        else:
            return loader_class(**loader_kwargs)

    @classmethod
    def load_documents(
        cls,
        loader_type: str,
        file_path: str,
        **kwargs
    ) -> List[Document]:
        """
        加载文档
        
        Args:
            loader_type: 加载器类型
            file_path: 文件路径
            **kwargs: 额外参数
            
        Returns:
            文档列表
        """
        loader = cls.create_loader(loader_type, file_path, **kwargs)
        documents = loader.load()
        app_logger.info(f"加载文档完成: {file_path}, 共 {len(documents)} 个文档块")
        return documents

    @classmethod
    def load_from_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: str = "pdf",
        **kwargs
    ) -> List[Document]:
        """
        从目录加载文档
        
        Args:
            directory_path: 目录路径
            glob_pattern: 文件匹配模式
            loader_type: 加载器类型
            **kwargs: 额外参数
            
        Returns:
            文档列表
        """
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_kwargs=kwargs,
            show_progress=True
        )
        documents = loader.load()
        app_logger.info(f"从目录加载文档完成: {directory_path}, 共 {len(documents)} 个文档块")
        return documents

    @classmethod
    def list_supported_loaders(cls) -> List[str]:
        """
        列出所有支持的加载器类型
        
        Returns:
            加载器类型列表
        """
        all_loaders = set(cls._builtin_loaders.keys())
        all_loaders.update(cls._custom_loaders.keys())
        all_loaders.update(rag_settings.loaders.keys())
        return list(all_loaders)

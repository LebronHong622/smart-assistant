"""
LangChain 文档加载器工厂
"""
from typing import Any, Dict, List, Optional, Type
from importlib import import_module
from config.rag_settings import rag_settings
from domain.shared.ports import LoaderFactoryPort
from domain.entity.document.document import Document
from infrastructure.rag.shared.converters import convert_lc_to_domain
from infrastructure.core.log import app_logger
from infrastructure.rag.document_loader.langchain.adapters import _LOADER_ADAPTERS


class LangChainLoaderFactory(LoaderFactoryPort):
    """
    LangChain 文档加载器工厂
    """

    _builtin_loaders: Dict[str, Type[Any]] = {}
    _custom_loaders: Dict[str, Type[Any]] = {}

    @classmethod
    def _ensure_builtin_loaders(cls):
        """延迟加载内置加载器"""
        if not cls._builtin_loaders:
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
            cls._builtin_loaders = {
                "pdf": PyPDFLoader,
                "text": TextLoader,
                "csv": CSVLoader,
                "json": JSONLoader,
                "directory": DirectoryLoader,
                "web": WebBaseLoader,
                "markdown": UnstructuredMarkdownLoader,
                "image": UnstructuredImageLoader,
            }

    @classmethod
    def register_loader(cls, name: str, loader_class: Type[Any]) -> None:
        """注册自定义加载器"""
        cls._custom_loaders[name] = loader_class
        app_logger.info(f"注册自定义文档加载器: {name}")

    @classmethod
    def get_loader_class(cls, loader_type: str) -> Optional[Type[Any]]:
        """获取加载器类"""
        cls._ensure_builtin_loaders()

        if loader_type in cls._custom_loaders:
            return cls._custom_loaders[loader_type]

        if loader_type in cls._builtin_loaders:
            return cls._builtin_loaders[loader_type]

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
    ) -> Any:
        """创建文档加载器实例"""
        config = rag_settings.get_loader_config(loader_type)
        if not config or not config.enabled:
            raise ValueError(f"不支持的文档加载器类型: {loader_type}")

        loader_kwargs = {**config.config, **kwargs}
        loader_class = cls.get_loader_class(loader_type)
        if not loader_class:
            raise ValueError(f"找不到文档加载器类: {loader_type}")

        app_logger.info(f"创建文档加载器: {loader_type}, 文件: {file_path}")

        # 使用适配器处理特殊配置（解耦点）
        adapter = _LOADER_ADAPTERS.get(loader_type)
        if adapter:
            loader_kwargs = adapter(file_path, loader_kwargs)

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
        """加载文档，返回领域 Document"""
        loader = cls.create_loader(loader_type, file_path, **kwargs)
        lc_docs = loader.load()
        app_logger.info(f"加载文档完成: {file_path}, 共 {len(lc_docs)} 个文档块")
        # 转换为领域 Document
        return convert_lc_to_domain(lc_docs)

    @classmethod
    def load_from_directory(
        cls,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        loader_type: str = "pdf",
        **kwargs
    ) -> List[Document]:
        """从目录加载文档，返回领域 Document"""
        from langchain_community.document_loaders import DirectoryLoader
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_kwargs=kwargs,
            show_progress=True
        )
        lc_docs = loader.load()
        app_logger.info(f"从目录加载文档完成: {directory_path}, 共 {len(lc_docs)} 个文档块")
        # 转换为领域 Document
        return convert_lc_to_domain(lc_docs)

    @classmethod
    def list_supported_loaders(cls) -> List[str]:
        """列出所有支持的加载器类型"""
        cls._ensure_builtin_loaders()
        all_loaders = set(cls._builtin_loaders.keys())
        all_loaders.update(cls._custom_loaders.keys())
        all_loaders.update(rag_settings.loaders.keys())
        return list(all_loaders)

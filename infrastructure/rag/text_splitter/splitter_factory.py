"""
文本分块器工厂
根据配置动态创建文本分块器实例
"""

from typing import Any, Dict, List, Optional, Type, Callable
from importlib import import_module

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
)
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter

from config.rag_settings import rag_settings, SplitterItemConfig
from infrastructure.core.log import app_logger


class TextSplitterFactory:
    """
    文本分块器工厂类
    
    根据配置动态创建不同类型的文本分块器
    支持扩展新的分块器类型
    """

    # 内置分块器映射
    _builtin_splitters: Dict[str, Type[TextSplitter]] = {
        "recursive": RecursiveCharacterTextSplitter,
        "character": CharacterTextSplitter,
        "token": TokenTextSplitter,
        "markdown": MarkdownTextSplitter,
    }

    # 自定义分块器注册表
    _custom_splitters: Dict[str, Type[TextSplitter]] = {}

    @classmethod
    def register_splitter(cls, name: str, splitter_class: Type[TextSplitter]) -> None:
        """
        注册自定义分块器
        
        Args:
            name: 分块器名称
            splitter_class: 分块器类
        """
        cls._custom_splitters[name] = splitter_class
        app_logger.info(f"注册自定义文本分块器: {name}")

    @classmethod
    def get_splitter_class(cls, splitter_type: str) -> Optional[Type[TextSplitter]]:
        """
        获取分块器类
        
        Args:
            splitter_type: 分块器类型
            
        Returns:
            分块器类或 None
        """
        # 先查找自定义分块器
        if splitter_type in cls._custom_splitters:
            return cls._custom_splitters[splitter_type]
        
        # 再查找内置分块器
        if splitter_type in cls._builtin_splitters:
            return cls._builtin_splitters[splitter_type]
        
        # 最后尝试从配置动态加载
        config = rag_settings.get_splitter_config(splitter_type)
        if config and config.enabled:
            try:
                module = import_module(config.module)
                splitter_class = getattr(module, config.class_name)
                cls._custom_splitters[splitter_type] = splitter_class
                return splitter_class
            except (ImportError, AttributeError) as e:
                app_logger.error(f"加载文本分块器失败 [{splitter_type}]: {e}")
        
        return None

    @classmethod
    def create_splitter(
        cls,
        splitter_type: str,
        **kwargs
    ) -> TextSplitter:
        """
        创建文本分块器实例
        
        Args:
            splitter_type: 分块器类型
            **kwargs: 额外参数，会覆盖配置文件中的默认值
            
        Returns:
            分块器实例
            
        Raises:
            ValueError: 不支持的分块器类型
        """
        # 获取配置
        config = rag_settings.get_splitter_config(splitter_type)
        if not config or not config.enabled:
            raise ValueError(f"不支持的文本分块器类型: {splitter_type}")

        # 合并配置参数（kwargs 优先）
        splitter_kwargs = {**config.config, **kwargs}
        
        # 获取分块器类
        splitter_class = cls.get_splitter_class(splitter_type)
        if not splitter_class:
            raise ValueError(f"找不到文本分块器类: {splitter_type}")

        app_logger.info(f"创建文本分块器: {splitter_type}, 参数: {splitter_kwargs}")

        return splitter_class(**splitter_kwargs)

    @classmethod
    def split_documents(
        cls,
        documents: List[Document],
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[Document]:
        """
        分块文档
        
        Args:
            documents: 待分块的文档列表
            splitter_type: 分块器类型
            **kwargs: 额外参数
            
        Returns:
            分块后的文档列表
        """
        splitter = cls.create_splitter(splitter_type, **kwargs)
        split_docs = splitter.split_documents(documents)
        app_logger.info(f"文档分块完成: 原始 {len(documents)} 个 -> 分块后 {len(split_docs)} 个")
        return split_docs

    @classmethod
    def split_text(
        cls,
        text: str,
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[str]:
        """
        分块文本
        
        Args:
            text: 待分块的文本
            splitter_type: 分块器类型
            **kwargs: 额外参数
            
        Returns:
            分块后的文本列表
        """
        splitter = cls.create_splitter(splitter_type, **kwargs)
        return splitter.split_text(text)

    @classmethod
    def create_code_splitter(
        cls,
        language: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> TextSplitter:
        """
        创建代码分块器
        
        Args:
            language: 编程语言
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            代码分块器实例
        """
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @classmethod
    def list_supported_splitters(cls) -> List[str]:
        """
        列出所有支持的分块器类型
        
        Returns:
            分块器类型列表
        """
        all_splitters = set(cls._builtin_splitters.keys())
        all_splitters.update(cls._custom_splitters.keys())
        all_splitters.update(rag_settings.splitters.keys())
        return list(all_splitters)

"""
LangChain 文本分割器工厂
"""
from typing import Any, Dict, List, Optional, Type
from importlib import import_module
from langchain_text_splitters import TextSplitter
from config.rag_settings import rag_settings
from domain.shared.ports import SplitterFactoryPort
from domain.entity.document.document import Document
from infrastructure.rag.shared.converters import (
    convert_lc_to_domain,
    convert_domain_to_lc,
)
from infrastructure.core.log import app_logger


class LangChainSplitterFactory(SplitterFactoryPort):
    """
    LangChain 文本分割器工厂
    """

    _builtin_splitters: Dict[str, Type[TextSplitter]] = {}
    _custom_splitters: Dict[str, Type[TextSplitter]] = {}

    @classmethod
    def _ensure_builtin_splitters(cls):
        """延迟加载内置分割器"""
        if not cls._builtin_splitters:
            from langchain_text_splitters import (
                RecursiveCharacterTextSplitter,
                CharacterTextSplitter,
                TokenTextSplitter,
                MarkdownTextSplitter,
            )
            cls._builtin_splitters = {
                "recursive": RecursiveCharacterTextSplitter,
                "character": CharacterTextSplitter,
                "token": TokenTextSplitter,
                "markdown": MarkdownTextSplitter,
            }

    @classmethod
    def register_splitter(cls, name: str, splitter_class: Type[TextSplitter]) -> None:
        """注册自定义分割器"""
        cls._custom_splitters[name] = splitter_class
        app_logger.info(f"注册自定义文本分割器: {name}")

    @classmethod
    def get_splitter_class(cls, splitter_type: str) -> Optional[Type[TextSplitter]]:
        """获取分割器类"""
        cls._ensure_builtin_splitters()

        if splitter_type in cls._custom_splitters:
            return cls._custom_splitters[splitter_type]

        if splitter_type in cls._builtin_splitters:
            return cls._builtin_splitters[splitter_type]

        config = rag_settings.get_splitter_config(splitter_type)
        if config and config.enabled:
            try:
                module = import_module(config.module)
                splitter_class = getattr(module, config.class_name)
                cls._custom_splitters[splitter_type] = splitter_class
                return splitter_class
            except (ImportError, AttributeError) as e:
                app_logger.error(f"加载文本分割器失败 [{splitter_type}]: {e}")

        return None

    @classmethod
    def create_splitter(
        cls,
        splitter_type: str = "recursive",
        **kwargs
    ) -> TextSplitter:
        """创建文本分割器实例"""
        config = rag_settings.get_splitter_config(splitter_type)
        if not config or not config.enabled:
            raise ValueError(f"不支持的文本分割器类型: {splitter_type}")

        splitter_kwargs = {**config.config, **kwargs}
        splitter_class = cls.get_splitter_class(splitter_type)
        if not splitter_class:
            raise ValueError(f"找不到文本分割器类: {splitter_type}")

        app_logger.info(f"创建文本分割器: {splitter_type}, 参数: {splitter_kwargs}")

        return splitter_class(**splitter_kwargs)

    @classmethod
    def split_documents(
        cls,
        documents: List[Document],
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[Document]:
        """分割文档，接受并返回领域 Document"""
        # 转换为 LangChain Document
        lc_docs = convert_domain_to_lc(documents)

        splitter = cls.create_splitter(splitter_type, **kwargs)
        split_lc_docs = splitter.split_documents(lc_docs)
        app_logger.info(f"文档分割完成: 原始 {len(documents)} 个 -> 分割后 {len(split_lc_docs)} 个")

        # 转换回领域 Document
        return convert_lc_to_domain(split_lc_docs)

    @classmethod
    def split_text(
        cls,
        text: str,
        splitter_type: str = "recursive",
        **kwargs
    ) -> List[str]:
        """分割文本"""
        splitter = cls.create_splitter(splitter_type, **kwargs)
        return splitter.split_text(text)

    @classmethod
    def create_code_splitter(
        cls,
        language: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> TextSplitter:
        """创建代码分割器"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @classmethod
    def list_supported_splitters(cls) -> List[str]:
        """列出所有支持的分割器类型"""
        cls._ensure_builtin_splitters()
        all_splitters = set(cls._builtin_splitters.keys())
        all_splitters.update(cls._custom_splitters.keys())
        all_splitters.update(rag_settings.splitters.keys())
        return list(all_splitters)

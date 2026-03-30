"""
文档分块领域服务
提供纯业务逻辑的文档分块策略和规则
"""

from typing import List
from pydantic import BaseModel
from enum import Enum


class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    CHARACTER = "character"  # 基于字符数的分块
    RECURSIVE_CHARACTER = "recursive_character"  # 递归字符分块（推荐，按段落、句子、字符递归）
    TOKEN = "token"  # 基于token的分块


class ChunkingConfig(BaseModel):
    """
    文档分块配置值对象
    """
    chunk_size: int = 500  # 每个chunk的大小（字符数或token数）
    chunk_overlap: int = 50  # chunk之间的重叠大小
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER
    separators: List[str] = None  # 自定义分隔符（仅用于递归分块）

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        if self.separators is None:
            # 默认递归分块的分隔符：先按段落分，再按句子分，最后按字符分
            object.__setattr__(self, 'separators', ["\n\n", "\n", "。", "!", "?", "；", "…", " ", ""])


class DocumentChunkingService:
    """
    文档分块领域服务
    负责文档分块的纯业务逻辑，不涉及文件I/O和基础设施
    """

    def __init__(self):
        pass

    def chunk_document(self, content: str, config: ChunkingConfig) -> List[str]:
        """
        根据配置对文档内容进行分块

        Args:
            content: 文档内容
            config: 分块配置

        Returns:
            分块后的文本列表
        """
        if not content:
            return []

        if config.strategy == ChunkingStrategy.CHARACTER:
            return self._chunk_by_character(content, config)
        elif config.strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
            return self._chunk_by_recursive_character(content, config)
        elif config.strategy == ChunkingStrategy.TOKEN:
            return self._chunk_by_token(content, config)
        else:
            raise ValueError(f"不支持的分块策略: {config.strategy}")

    def validate_chunk(self, chunk: str, config: ChunkingConfig) -> bool:
        """
        验证分块是否符合配置要求

        Args:
            chunk: 分块文本
            config: 分块配置

        Returns:
            是否有效
        """
        if not chunk.strip():
            return False

        # 检查分块大小是否合理（允许一定的偏差，但不应该超过配置的1.5倍）
        chunk_length = len(chunk)
        return chunk_length <= config.chunk_size * 1.5

    def estimate_chunks_count(self, content: str, config: ChunkingConfig) -> int:
        """
        估算文档会被分成多少个chunk

        Args:
            content: 文档内容
            config: 分块配置

        Returns:
            估算的chunk数量
        """
        if not content:
            return 0

        effective_chunk_size = config.chunk_size - config.chunk_overlap
        if effective_chunk_size <= 0:
            effective_chunk_size = 1

        content_length = len(content)
        estimated_chunks = (content_length + effective_chunk_size - 1) // effective_chunk_size

        return estimated_chunks

    def _chunk_by_character(self, content: str, config: ChunkingConfig) -> List[str]:
        """基于字符数的简单分块"""
        chunks = []
        start = 0
        content_length = len(content)

        while start < content_length:
            end = start + config.chunk_size
            chunk = content[start:end]

            if chunk:
                chunks.append(chunk)

            # 计算下一个chunk的起始位置（考虑重叠）
            start = end - config.chunk_overlap
            if start <= 0:
                start = end

        return chunks

    def _chunk_by_recursive_character(self, content: str, config: ChunkingConfig) -> List[str]:
        """递归字符分块（推荐方式）"""
        chunks = []

        def _recursive_split(text: str, separators: List[str]) -> List[str]:
            """递归分割文本"""
            if len(text) <= config.chunk_size:
                return [text] if text else []

            # 按分隔符分割
            for separator in separators:
                if separator not in text:
                    continue

                parts = text.split(separator)
                current_chunks = []
                current_chunk = ""

                for part in parts:
                    test_chunk = current_chunk + part if current_chunk else part

                    if len(test_chunk) <= config.chunk_size:
                        current_chunk = test_chunk + separator if separator else test_chunk
                    else:
                        if current_chunk:
                            current_chunks.append(current_chunk.rstrip())
                        current_chunk = part + separator if separator else part

                if current_chunk:
                    current_chunks.append(current_chunk.rstrip())

                # 如果成功分割出多个chunk，返回结果
                if len(current_chunks) > 1:
                    # 处理重叠
                    final_chunks = []
                    for i, chunk in enumerate(current_chunks):
                        final_chunks.append(chunk)
                        # 添加重叠部分到下一个chunk
                        if i < len(current_chunks) - 1 and config.chunk_overlap > 0:
                            overlap_text = chunk[-config.chunk_overlap:]
                            current_chunks[i + 1] = overlap_text + current_chunks[i + 1]
                    return final_chunks

            # 如果所有分隔符都无法分割，使用字符级分割
            return [text[i:i + config.chunk_size] for i in range(0, len(text), config.chunk_size - config.chunk_overlap)]

        chunks = _recursive_split(content, config.separators)
        return [c for c in chunks if c.strip()]

    def _chunk_by_token(self, content: str, config: ChunkingConfig) -> List[str]:
        """
        基于token的分块
        注意：这是一个简化实现，实际应该使用分词器（如tiktoken）
        """
        # 这里使用字符估算作为token的近似（中文字符通常1字符≈1token）
        # 实际项目中应该使用真实的tokenizer
        return self._chunk_by_character(content, config)

    def get_default_config(self) -> ChunkingConfig:
        """
        获取默认的分块配置

        Returns:
            默认分块配置
        """
        return ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER
        )

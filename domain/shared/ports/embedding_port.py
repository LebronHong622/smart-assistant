"""
嵌入向量生成端口 - 定义嵌入向量接口
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingGeneratorPort(ABC):
    """嵌入向量生成接口"""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        pass

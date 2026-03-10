"""
嵌入向量适配器 - 实现嵌入向量端口
"""

from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from infrastructure.external.model.embeddings_manager import EmbeddingsManager


class EmbeddingAdapter(EmbeddingGeneratorPort):
    """嵌入向量适配器实现"""

    def __init__(self):
        self._embeddings_manager = EmbeddingsManager()

    def generate_embedding(self, text: str) -> list[float]:
        """生成单个文本的嵌入向量"""
        return self._embeddings_manager.generate_embedding(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的嵌入向量"""
        return self._embeddings_manager.generate_embeddings(texts)

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self._embeddings_manager.get_embedding_dimension()

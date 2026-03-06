"""
Embeddings 模型管理
使用阿里千文（DashScope）text-embedding-v3 模型
"""

import dashscope
from dashscope import TextEmbedding
from infrastructure.config.settings import settings


class EmbeddingsManager:
    """Embeddings 模型管理器"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # 初始化 DashScope
            dashscope.api_key = settings.dashscope.dashscope_api_key
            self._initialized = True

    def generate_embedding(self, text: str) -> list[float]:
        """
        生成文本的嵌入向量

        Args:
            text: 要生成嵌入向量的文本

        Returns:
            嵌入向量（1536 维）
        """
        try:
            # 确保文本长度适中
            text = self._truncate_text(text)

            response = TextEmbedding.call(
                model=settings.dashscope.dashscope_embedding_model,
                input=text
            )

            if response.status_code != 200:
                raise RuntimeError(f"生成文本嵌入向量失败: {response}")

            return response.output["embeddings"][0]["embedding"]

        except Exception as e:
            raise RuntimeError(f"生成文本嵌入向量失败: {str(e)}")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本的嵌入向量

        Args:
            texts: 要生成嵌入向量的文本列表

        Returns:
            嵌入向量列表
        """
        try:
            # 确保文本长度适中
            truncated_texts = [self._truncate_text(text) for text in texts]

            response = TextEmbedding.call(
                model=settings.dashscope.dashscope_embedding_model,
                input=truncated_texts
            )

            if response.status_code != 200:
                raise RuntimeError(f"批量生成文本嵌入向量失败: {response}")

            return [item["embedding"] for item in response.output["embeddings"]]

        except Exception as e:
            raise RuntimeError(f"批量生成文本嵌入向量失败: {str(e)}")

    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        截断文本以避免超过模型的最大输入限制

        Args:
            text: 要截断的文本
            max_tokens: 最大令牌数（大致估算）

        Returns:
            截断后的文本
        """
        # 简单的截断策略：按字符数估算
        if len(text) > max_tokens * 2:
            return text[:max_tokens * 2]
        return text

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return 1536
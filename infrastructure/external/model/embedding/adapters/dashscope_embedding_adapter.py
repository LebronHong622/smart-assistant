"""
DashScope 嵌入向量适配器 - 实现嵌入向量端口
"""

import dashscope
from dashscope import TextEmbedding
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from config.settings import settings


class DashScopeEmbeddingAdapter(EmbeddingGeneratorPort):
    """DashScope 嵌入向量适配器实现"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            dashscope.api_key = settings.dashscope.dashscope_api_key
            self._initialized = True

    def generate_embedding(self, text: str) -> list[float]:
        """生成文本的嵌入向量"""
        try:
            text = self._truncate_text(text)
            response = TextEmbedding.call(
                model=settings.dashscope.dashscope_embedding_model,
                input=text,
                dimension=settings.dashscope.dashscope_embedding_dim
            )
            if response.status_code != 200:
                raise RuntimeError(f"生成文本嵌入向量失败: {response}")
            return response.output["embeddings"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"生成文本嵌入向量失败: {str(e)}")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本的嵌入向量"""
        try:
            truncated_texts = [self._truncate_text(text) for text in texts]
            response = TextEmbedding.call(
                model=settings.dashscope.dashscope_embedding_model,
                input=truncated_texts,
                parameters={
                    "text_type": "document",
                    "dimensions": settings.dashscope.dashscope_embedding_dim
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f"批量生成文本嵌入向量失败: {response}")
            return [item["embedding"] for item in response.output["embeddings"]]
        except Exception as e:
            raise RuntimeError(f"批量生成文本嵌入向量失败: {str(e)}")

    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """截断文本以避免超过模型的最大输入限制"""
        if len(text) > max_tokens * 2:
            return text[:max_tokens * 2]
        return text

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return settings.dashscope.dashscope_embedding_dim

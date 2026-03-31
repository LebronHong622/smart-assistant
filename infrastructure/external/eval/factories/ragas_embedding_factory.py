"""
Ragas Embedding工厂
从YAML配置创建符合ragas BaseRagasEmbeddings接口的Embedding实例
client参数使用OpenAI(...)进行初始化
"""
from openai import OpenAI
from ragas.embeddings.base import BaseRagasEmbeddings, embedding_factory
from config.eval_settings import EmbeddingConfig


class RagasEmbeddingFactory:
    """
    Ragas Embedding 工厂
    支持dashscope/openai/deepseek，都使用OpenAI兼容模式
    """

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> BaseRagasEmbeddings:
        """根据配置创建Embedding实例

        Args:
            config: Embedding配置

        Returns:
            符合ragas BaseRagasEmbeddings接口的实例
        """
        # 先创建OpenAI client
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

        if config.provider in ["dashscope", "openai", "deepseek"]:
            # 都是OpenAI兼容模式
            return embedding_factory(
                model=config.model_name,
                client=client,
            )
        else:
            raise ValueError(f"不支持的Embedding提供商: {config.provider}")

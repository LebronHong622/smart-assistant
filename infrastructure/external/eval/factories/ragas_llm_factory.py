"""
Ragas LLM工厂
从YAML配置创建符合ragas BaseRagasLLM接口的LLM实例
client参数使用OpenAI(...)进行初始化
"""
from openai import OpenAI
from ragas.llms.base import BaseRagasLLM
from ragas.llms.factory import llm_factory
from domain.eval.test_dataset_config import LLMConfig


class RagasLLMFactory:
    """
    Ragas LLM 工厂
    支持deepseek/dashscope/openai，都使用OpenAI兼容模式
    """

    @classmethod
    def from_config(cls, config: LLMConfig) -> BaseRagasLLM:
        """根据配置创建LLM实例

        Args:
            config: LLM配置

        Returns:
            符合ragas BaseRagasLLM接口的实例
        """
        # 先创建OpenAI client
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

        # 使用ragas llm_factory，传入client参数
        if config.provider in ["deepseek", "dashscope", "openai"]:
            # 都是OpenAI兼容模式
            return llm_factory(
                model="openai",
                client=client,
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")

"""
Ragas LLM工厂
从YAML配置创建符合ragas BaseRagasLLM接口的LLM实例
client参数使用OpenAI(...)进行初始化
"""
from openai import OpenAI
import instructor
from ragas.llms.base import BaseRagasLLM, InstructorLLM
from config.eval_settings import LLMConfig


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
        # 创建OpenAI client并使用instructor打补丁
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        client = instructor.from_openai(client)

        # 直接创建InstructorLLM
        if config.provider in ["deepseek", "dashscope", "openai"]:
            return InstructorLLM(
                client=client,
                model=config.model_name,
                provider="openai",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        else:
            raise ValueError(f"不支持的LLM提供商: {config.provider}")

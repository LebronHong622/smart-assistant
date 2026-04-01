"""
Ragas LLM工厂单元测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from ragas.llms.base import BaseRagasLLM

from infrastructure.external.eval.factories.ragas_llm_factory import RagasLLMFactory
from config.eval_settings import LLMConfig


class TestRagasLLMFactory:
    """RagasLLMFactory测试类"""

    def test_from_config_openai_provider(self):
        """测试OpenAI provider配置"""
        config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
            max_tokens=1000
        )

        with patch('infrastructure.external.eval.factories.ragas_llm_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_llm_factory.llm_factory') as mock_factory:
                mock_llm = Mock(spec=BaseRagasLLM)
                mock_factory.return_value = mock_llm
                
                result = RagasLLMFactory.from_config(config)
                
                # 验证OpenAI client创建
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url="https://api.openai.com/v1"
                )
                
                # 验证llm_factory调用
                mock_factory.assert_called_once_with(
                    model="openai",
                    client=mock_client,
                    model_name="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=1000
                )
                
                assert result == mock_llm

    def test_from_config_deepseek_provider(self):
        """测试DeepSeek provider配置"""
        config = LLMConfig(
            provider="deepseek",
            model_name="deepseek-chat",
            api_key="test-api-key",
            base_url="https://api.deepseek.com/v1",
            temperature=0.5,
            max_tokens=2000
        )

        with patch('infrastructure.external.eval.factories.ragas_llm_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_llm_factory.llm_factory') as mock_factory:
                mock_llm = Mock(spec=BaseRagasLLM)
                mock_factory.return_value = mock_llm
                
                result = RagasLLMFactory.from_config(config)
                
                # 验证OpenAI client创建
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url="https://api.deepseek.com/v1"
                )
                
                # 验证llm_factory调用
                mock_factory.assert_called_once_with(
                    model="openai",
                    client=mock_client,
                    model_name="deepseek-chat",
                    temperature=0.5,
                    max_tokens=2000
                )
                
                assert result == mock_llm

    def test_from_config_dashscope_provider(self):
        """测试DashScope provider配置"""
        config = LLMConfig(
            provider="dashscope",
            model_name="qwen-turbo",
            api_key="test-api-key",
            base_url="https://dashscope.aliyuncs.com/api/v1",
            temperature=0.3,
            max_tokens=1500
        )

        with patch('infrastructure.external.eval.factories.ragas_llm_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_llm_factory.llm_factory') as mock_factory:
                mock_llm = Mock(spec=BaseRagasLLM)
                mock_factory.return_value = mock_llm
                
                result = RagasLLMFactory.from_config(config)
                
                # 验证OpenAI client创建
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url="https://dashscope.aliyuncs.com/api/v1"
                )
                
                # 验证llm_factory调用
                mock_factory.assert_called_once_with(
                    model="openai",
                    client=mock_client,
                    model_name="qwen-turbo",
                    temperature=0.3,
                    max_tokens=1500
                )
                
                assert result == mock_llm

    def test_from_config_unsupported_provider(self):
        """测试不支持的provider抛出异常"""
        config = LLMConfig(
            provider="unsupported_provider",
            model_name="test-model",
            api_key="test-api-key",
            base_url="https://api.test.com/v1",
            temperature=0.7,
            max_tokens=1000
        )

        with pytest.raises(ValueError, match="不支持的LLM提供商: unsupported_provider"):
            RagasLLMFactory.from_config(config)

    def test_from_config_default_temperature_max_tokens(self):
        """测试默认temperature和max_tokens值"""
        config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1"
            # 没有指定temperature和max_tokens，应该使用默认值
        )

        with patch('infrastructure.external.eval.factories.ragas_llm_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_llm_factory.llm_factory') as mock_factory:
                mock_llm = Mock(spec=BaseRagasLLM)
                mock_factory.return_value = mock_llm
                
                result = RagasLLMFactory.from_config(config)
                
                # 验证llm_factory调用时使用了默认的temperature和max_tokens
                mock_factory.assert_called_once_with(
                    model="openai",
                    client=mock_client,
                    model_name="gpt-3.5-turbo",
                    temperature=config.temperature,  # 应该是默认值
                    max_tokens=config.max_tokens  # 应该是默认值
                )
                
                assert result == mock_llm

    def test_from_config_empty_base_url(self):
        """测试base_url为空字符串的情况"""
        config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            base_url="",
            temperature=0.7,
            max_tokens=1000
        )

        with patch('infrastructure.external.eval.factories.ragas_llm_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_llm_factory.llm_factory') as mock_factory:
                mock_llm = Mock(spec=BaseRagasLLM)
                mock_factory.return_value = mock_llm
                
                result = RagasLLMFactory.from_config(config)
                
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url=""
                )
                
                assert result == mock_llm

    def test_from_config_zero_temperature(self):
        """测试temperature为0的情况"""
        config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=1000
        )

        with patch('infrastructure.external.eval.factories.ragas_llm_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_llm_factory.llm_factory') as mock_factory:
                mock_llm = Mock(spec=BaseRagasLLM)
                mock_factory.return_value = mock_llm
                
                result = RagasLLMFactory.from_config(config)
                
                mock_factory.assert_called_once_with(
                    model="openai",
                    client=mock_client,
                    model_name="gpt-3.5-turbo",
                    temperature=0.0,
                    max_tokens=1000
                )
                
                assert result == mock_llm
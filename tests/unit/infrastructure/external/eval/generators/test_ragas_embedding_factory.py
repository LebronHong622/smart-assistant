"""
Ragas Embedding工厂单元测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from ragas.embeddings.base import BaseRagasEmbeddings

from infrastructure.external.eval.factories.ragas_embedding_factory import RagasEmbeddingFactory
from config.eval_settings import EmbeddingConfig


class TestRagasEmbeddingFactory:
    """RagasEmbeddingFactory测试类"""

    def test_from_config_openai_provider(self):
        """测试OpenAI provider配置"""
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1"
        )

        with patch('infrastructure.external.eval.factories.ragas_embedding_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_embedding_factory.embedding_factory') as mock_factory:
                mock_embedding = Mock(spec=BaseRagasEmbeddings)
                mock_factory.return_value = mock_embedding
                
                result = RagasEmbeddingFactory.from_config(config)
                
                # 验证OpenAI client创建
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url="https://api.openai.com/v1"
                )
                
                # 验证embedding_factory调用
                mock_factory.assert_called_once_with(
                    model="text-embedding-ada-002",
                    client=mock_client
                )
                
                assert result == mock_embedding

    def test_from_config_dashscope_provider(self):
        """测试DashScope provider配置"""
        config = EmbeddingConfig(
            provider="dashscope",
            model_name="text-embedding-v1",
            api_key="test-api-key",
            base_url="https://dashscope.aliyuncs.com/api/v1"
        )

        with patch('infrastructure.external.eval.factories.ragas_embedding_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_embedding_factory.embedding_factory') as mock_factory:
                mock_embedding = Mock(spec=BaseRagasEmbeddings)
                mock_factory.return_value = mock_embedding
                
                result = RagasEmbeddingFactory.from_config(config)
                
                # 验证OpenAI client创建
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url="https://dashscope.aliyuncs.com/api/v1"
                )
                
                # 验证embedding_factory调用
                mock_factory.assert_called_once_with(
                    model="text-embedding-v1",
                    client=mock_client
                )
                
                assert result == mock_embedding

    def test_from_config_deepseek_provider(self):
        """测试DeepSeek provider配置"""
        config = EmbeddingConfig(
            provider="deepseek",
            model_name="deepseek-embedding",
            api_key="test-api-key",
            base_url="https://api.deepseek.com/v1"
        )

        with patch('infrastructure.external.eval.factories.ragas_embedding_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_embedding_factory.embedding_factory') as mock_factory:
                mock_embedding = Mock(spec=BaseRagasEmbeddings)
                mock_factory.return_value = mock_embedding
                
                result = RagasEmbeddingFactory.from_config(config)
                
                # 验证OpenAI client创建
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url="https://api.deepseek.com/v1"
                )
                
                # 验证embedding_factory调用
                mock_factory.assert_called_once_with(
                    model="deepseek-embedding",
                    client=mock_client
                )
                
                assert result == mock_embedding

    def test_from_config_unsupported_provider(self):
        """测试不支持的provider抛出异常"""
        config = EmbeddingConfig(
            provider="unsupported_provider",
            model_name="test-model",
            api_key="test-api-key",
            base_url="https://api.test.com/v1"
        )

        with pytest.raises(ValueError, match="不支持的Embedding提供商: unsupported_provider"):
            RagasEmbeddingFactory.from_config(config)

    def test_from_config_missing_api_key(self):
        """测试缺少API key的情况"""
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            api_key="",
            base_url="https://api.openai.com/v1"
        )

        with patch('infrastructure.external.eval.factories.ragas_embedding_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_embedding_factory.embedding_factory') as mock_factory:
                mock_embedding = Mock(spec=BaseRagasEmbeddings)
                mock_factory.return_value = mock_embedding
                
                # 应该正常执行，不验证API key有效性
                result = RagasEmbeddingFactory.from_config(config)
                
                mock_openai.assert_called_once_with(
                    api_key="",
                    base_url="https://api.openai.com/v1"
                )
                
                assert result == mock_embedding

    def test_from_config_empty_base_url(self):
        """测试base_url为空字符串的情况"""
        config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-ada-002",
            api_key="test-api-key",
            base_url=""
        )

        with patch('infrastructure.external.eval.factories.ragas_embedding_factory.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch('infrastructure.external.eval.factories.ragas_embedding_factory.embedding_factory') as mock_factory:
                mock_embedding = Mock(spec=BaseRagasEmbeddings)
                mock_factory.return_value = mock_embedding
                
                result = RagasEmbeddingFactory.from_config(config)
                
                mock_openai.assert_called_once_with(
                    api_key="test-api-key",
                    base_url=""
                )
                
                assert result == mock_embedding
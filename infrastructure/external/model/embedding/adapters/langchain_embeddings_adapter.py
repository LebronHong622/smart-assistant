"""
LangChain Embeddings 适配器
将现有 EmbeddingGeneratorPort 实现包装为 LangChain 标准 Embeddings 接口
"""
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, PrivateAttr
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from infrastructure.external.model.embedding.factories.embedding_factory import EmbeddingFactory
from config.settings import get_app_settings

settings = get_app_settings()

class LangChainEmbeddingsAdapter(BaseModel, Embeddings):
    """
    LangChain Embeddings 适配器
    兼容 LangChain 标准接口，内部使用项目现有 EmbeddingGeneratorPort 实现
    """
    provider: str = Field(default_factory=lambda: settings.app.langchain_embeddings_provider)
    _embedding_generator: Optional[EmbeddingGeneratorPort] = PrivateAttr(default=None)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedding_generator = EmbeddingFactory.get(self.provider)

    def embed_query(self, text: str) -> List[float]:
        """
        生成查询文本的嵌入向量
        :param text: 查询文本
        :return: 嵌入向量
        """
        if not self._embedding_generator:
            raise RuntimeError("Embedding generator not initialized")
        return self._embedding_generator.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档的嵌入向量
        :param texts: 文档文本列表
        :return: 嵌入向量列表
        """
        if not self._embedding_generator:
            raise RuntimeError("Embedding generator not initialized")
        return self._embedding_generator.embed_texts(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        异步生成查询文本的嵌入向量
        :param text: 查询文本
        :return: 嵌入向量
        """
        if not self._embedding_generator:
            raise RuntimeError("Embedding generator not initialized")
        return await self._embedding_generator.aembed_text(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量生成文档的嵌入向量
        :param texts: 文档文本列表
        :return: 嵌入向量列表
        """
        if not self._embedding_generator:
            raise RuntimeError("Embedding generator not initialized")
        return await self._embedding_generator.aembed_texts(texts)

    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度
        :return: 向量维度
        """
        if not self._embedding_generator:
            raise RuntimeError("Embedding generator not initialized")
        return self._embedding_generator.get_embedding_dimension()

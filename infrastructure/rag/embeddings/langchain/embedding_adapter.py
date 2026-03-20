"""
LangChain Embedding 适配层
将 LangChain Embeddings 适配为领域 EmbeddingGeneratorPort 接口
"""
from typing import List
from langchain_core.embeddings import Embeddings
from domain.document.entity.document import Document
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort


class LangChainEmbeddingAdapter(EmbeddingGeneratorPort):
    """
    将 LangChain Embeddings 适配为领域 EmbeddingGeneratorPort

    实现领域接口，内部使用 LangChain Embeddings 进行实际的嵌入生成。
    """

    def __init__(self, embeddings: Embeddings, dimension: int = 768):
        self._embeddings = embeddings
        self._dimension = dimension

    # === 文本嵌入（同步）===

    def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        return self._embeddings.embed_documents(texts)

    # === 文档嵌入（同步）===

    def embed_document(self, document: Document) -> Document:
        """为单个文档生成嵌入向量，返回带嵌入的文档"""
        embedding = self.embed_text(document.content)
        document.embedding = embedding
        return document

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """批量生成文档的嵌入向量，返回带嵌入的文档列表"""
        texts = [doc.content for doc in documents]
        embeddings = self.embed_texts(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return documents

    # === 异步方法 ===

    async def aembed_text(self, text: str) -> List[float]:
        """异步生成单个文本的嵌入向量"""
        return await self._embeddings.aembed_query(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """异步批量生成文本的嵌入向量"""
        return await self._embeddings.aembed_documents(texts)

    async def aembed_document(self, document: Document) -> Document:
        """异步为单个文档生成嵌入向量"""
        embedding = await self.aembed_text(document.content)
        document.embedding = embedding
        return document

    async def aembed_documents(self, documents: List[Document]) -> List[Document]:
        """异步批量生成文档的嵌入向量"""
        texts = [doc.content for doc in documents]
        embeddings = await self.aembed_texts(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return documents

    # === 元信息 ===

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self._dimension

    # === LangChain 兼容接口（用于需要原始 Embeddings 的场景）===

    def to_langchain_embeddings(self) -> Embeddings:
        """获取原始 LangChain Embeddings 实例"""
        return self._embeddings

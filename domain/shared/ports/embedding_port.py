"""
嵌入向量生成端口 - 定义嵌入向量接口

支持：
- text (str) 和 Document 的嵌入生成
- 同步和异步方法
"""

from abc import ABC, abstractmethod
from typing import List

from domain.entity.document.document import Document


class EmbeddingGeneratorPort(ABC):
    """
    嵌入向量生成接口

    提供 text 和 Document 两类嵌入生成方法，支持同步和异步调用。
    """

    # === 文本嵌入（同步）===

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        pass

    # === 文档嵌入（同步）===

    @abstractmethod
    def embed_document(self, document: Document) -> Document:
        """为单个文档生成嵌入向量，返回带嵌入的文档"""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """批量生成文档的嵌入向量，返回带嵌入的文档列表"""
        pass

    # === 文本嵌入（异步）===

    async def aembed_text(self, text: str) -> List[float]:
        """异步生成单个文本的嵌入向量（默认同步实现）"""
        return self.embed_text(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """异步批量生成文本的嵌入向量（默认同步实现）"""
        return self.embed_texts(texts)

    # === 文档嵌入（异步）===

    async def aembed_document(self, document: Document) -> Document:
        """异步为单个文档生成嵌入向量（默认同步实现）"""
        return self.embed_document(document)

    async def aembed_documents(self, documents: List[Document]) -> List[Document]:
        """异步批量生成文档的嵌入向量（默认同步实现）"""
        return self.embed_documents(documents)

    # === 元信息 ===

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        pass

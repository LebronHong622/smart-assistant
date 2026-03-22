"""文档检索工具（包含标准实现和 LangChain 原生实现）"""
from .schema import (
    DocumentRetrievalInput,
    EmbeddingRetrievalInput,
    LangChainDocumentRetrievalInput
)
from .standard import document_retrieval, retrieve_similar_documents_by_embedding
from .langchain import langchain_document_retrieval

__all__ = [
    "DocumentRetrievalInput",
    "EmbeddingRetrievalInput",
    "LangChainDocumentRetrievalInput",
    "document_retrieval",
    "retrieve_similar_documents_by_embedding",
    "langchain_document_retrieval"
]

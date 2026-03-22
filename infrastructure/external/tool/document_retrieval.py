"""
文档检索工具
向后兼容：内容已移动到 tools/document_retrieval/
"""
from .tools.document_retrieval.standard import document_retrieval, retrieve_similar_documents_by_embedding

__all__ = ["document_retrieval", "retrieve_similar_documents_by_embedding"]

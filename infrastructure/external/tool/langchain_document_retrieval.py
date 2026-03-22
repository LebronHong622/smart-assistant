"""
基于 LangChain 的文档检索工具
向后兼容：内容已移动到 tools/document_retrieval/langchain.py
"""
from .tools.document_retrieval.langchain import langchain_document_retrieval

__all__ = ["langchain_document_retrieval"]

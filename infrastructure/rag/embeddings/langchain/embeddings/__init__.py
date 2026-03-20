"""
各个 Embedding Provider 实现
每个 provider 单独一个文件
"""
from infrastructure.rag.embeddings.langchain.embeddings.dashscope import create_dashscope_embedding
from infrastructure.rag.embeddings.langchain.embeddings.openai import create_openai_embedding
from infrastructure.rag.embeddings.langchain.embeddings.huggingface import create_huggingface_embedding

__all__ = [
    "create_dashscope_embedding",
    "create_openai_embedding",
    "create_huggingface_embedding",
]

"""
HuggingFace 嵌入实现
使用 sentence-transformers 模型
"""
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_huggingface_embedding(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Embeddings:
    """创建 HuggingFace 嵌入函数"""
    return HuggingFaceEmbeddings(model_name=model_name)

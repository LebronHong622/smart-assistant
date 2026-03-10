# embedding/adapters - 嵌入向量适配器
from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter
from infrastructure.external.model.embedding.adapters.langchain_embeddings_adapter import LangChainEmbeddingsAdapter

__all__ = [
    "DashScopeEmbeddingAdapter",
    "LangChainEmbeddingsAdapter"
]

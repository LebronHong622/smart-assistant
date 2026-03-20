"""
DashScope 嵌入实现
支持通过 dimension 参数指定输出向量维度（text-embedding-v3 支持 1024/768/512）
"""
from typing import Any, List, Optional
from pydantic import ConfigDict
from langchain_core.embeddings import Embeddings
from config.rag_settings import rag_settings
from config.settings import settings


def create_dashscope_embedding(
    model: Optional[str] = None,
    dimension: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Embeddings:
    """创建 DashScope 嵌入函数

    支持通过 dimension 参数指定输出向量维度（text-embedding-v3 支持 1024/768/512）
    """
    from langchain_community.embeddings import DashScopeEmbeddings

    config = rag_settings.get_embedding_config("dashscope")
    actual_model = model or (config.model if config else "text-embedding-v3")
    actual_dimension = dimension or (config.dimension if config else None)

    # 创建支持 dimension 参数的自定义 DashScope Embeddings
    class DashScopeEmbeddingsWithDimension(DashScopeEmbeddings):
        """支持 dimension 参数的 DashScope Embeddings"""
        dimension: Optional[int] = None

        model_config = ConfigDict(extra="allow")

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            kwargs = {"input": texts, "text_type": "document", "model": self.model}
            if self.dimension:
                kwargs["dimension"] = self.dimension
            result = self.client.call(**kwargs)
            if result.status_code == 200:
                return [item["embedding"] for item in result.output["embeddings"]]
            raise ValueError(f"Embedding failed: {result.message}")

        def embed_query(self, text: str) -> List[float]:
            kwargs = {"input": text, "text_type": "query", "model": self.model}
            if self.dimension:
                kwargs["dimension"] = self.dimension
            result = self.client.call(**kwargs)
            if result.status_code == 200:
                return result.output["embeddings"][0]["embedding"]
            raise ValueError(f"Embedding failed: {result.message}")

    return DashScopeEmbeddingsWithDimension(
        model=actual_model,
        dashscope_api_key=settings.dashscope.dashscope_api_key,
        dimension=actual_dimension,
    )

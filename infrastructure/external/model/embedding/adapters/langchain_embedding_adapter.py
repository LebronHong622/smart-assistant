from typing import List
from domain.shared.ports.model_capability_port import BaseModel, EmbeddingModelExtension
from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter
from infrastructure.core.log import app_logger


class LangChainEmbeddingAdapter(BaseModel[str, List[float]], EmbeddingModelExtension):
    """LangChain嵌入模型适配器，实现统一的模型调用接口"""

    def __init__(self):
        self._embedding_adapter = DashScopeEmbeddingAdapter()
        app_logger.info("初始化嵌入模型适配器")

    def invoke(self, input: str, **kwargs) -> List[float]:
        """同步生成单个文本的嵌入向量"""
        app_logger.debug(f"生成嵌入向量: input={input[:100]}...")
        return self._embedding_adapter.generate_embedding(input)

    async def ainvoke(self, input: str, **kwargs) -> List[float]:
        """异步生成单个文本的嵌入向量"""
        # DashScope当前没有原生异步接口，使用同步方法模拟
        app_logger.debug(f"异步生成嵌入向量: input={input[:100]}...")
        return self.invoke(input)

    def batch_invoke(self, inputs: List[str], **kwargs) -> List[List[float]]:
        """批量生成嵌入向量"""
        app_logger.debug(f"批量生成嵌入向量: count={len(inputs)}")
        return self._embedding_adapter.generate_embeddings(inputs)

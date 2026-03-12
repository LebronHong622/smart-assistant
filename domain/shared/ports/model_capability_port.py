from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Any

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class BaseModel(Generic[InputT, OutputT], ABC):
    """基础模型抽象接口，所有模型都实现这个统一接口"""

    @abstractmethod
    def invoke(self, input: InputT, **kwargs) -> OutputT:
        """同步调用模型"""
        pass

    @abstractmethod
    async def ainvoke(self, input: InputT, **kwargs) -> OutputT:
        """异步调用模型"""
        pass


class EmbeddingModelExtension(ABC):
    """嵌入模型扩展接口，提供批量处理能力"""

    @abstractmethod
    def batch_invoke(self, inputs: List[str], **kwargs) -> List[List[float]]:
        """批量生成嵌入向量"""
        pass

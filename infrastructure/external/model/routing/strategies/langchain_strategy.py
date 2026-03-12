from typing import Any
from domain.shared.model_enums import ModelType
from domain.shared.ports.model_router_port import BaseRoutingStrategy
from domain.shared.ports.model_capability_port import BaseModel
from infrastructure.external.model.llm.adapters.langchain_chat_adapter import DeepSeekChatModel
from infrastructure.external.model.embedding.adapters.langchain_embedding_adapter import LangChainEmbeddingAdapter
from infrastructure.core.log import app_logger


class LangChainChatStrategy(BaseRoutingStrategy):
    """LangChain普通聊天模型策略"""

    def select_model(
        self,
        model_type: ModelType,
        **kwargs
    ) -> BaseModel[Any, Any]:
        app_logger.debug(f"使用LangChainChatStrategy创建模型: type={model_type.value}")
        return DeepSeekChatModel(tool_enabled=False, **kwargs)


class LangChainToolChatStrategy(BaseRoutingStrategy):
    """LangChain带工具调用的聊天模型策略"""

    def select_model(
        self,
        model_type: ModelType,
        **kwargs
    ) -> BaseModel[Any, Any]:
        app_logger.debug(f"使用LangChainToolChatStrategy创建模型: type={model_type.value}")
        return DeepSeekChatModel(tool_enabled=True, **kwargs)


class LangChainEmbeddingStrategy(BaseRoutingStrategy):
    """LangChain嵌入模型策略"""

    def select_model(
        self,
        model_type: ModelType,
        **kwargs
    ) -> BaseModel[Any, Any]:
        app_logger.debug(f"使用LangChainEmbeddingStrategy创建模型: type={model_type.value}")
        return LangChainEmbeddingAdapter(**kwargs)
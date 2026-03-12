from typing import Any, Union, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models import LanguageModelInput
from domain.shared.ports.model_capability_port import BaseModel
from config.settings import settings
from infrastructure.core.log import app_logger


class BaseOpenAIChatModel(BaseModel[LanguageModelInput, Union[str, AIMessage]]):
    """OpenAI兼容聊天模型基类，实现通用功能，符合ChatOpenAI接口规范"""
    def __init__(self, api_key: str, base_url: str, model_name: str,
                 temperature: float = 0.7, max_tokens: int = 1024,
                 tool_enabled: bool = False):
        self.tool_enabled = tool_enabled
        self._model = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        app_logger.info(f"初始化OpenAI兼容模型: model={model_name}, tool_enabled={self.tool_enabled}")

    def invoke(self, input: LanguageModelInput, **kwargs) -> Union[str, AIMessage]:
        """
        同步调用模型，符合ChatOpenAI接口规范
        Args:
            input: LanguageModelInput类型，支持str、List[BaseMessage]等格式
        Returns:
            tool_enabled=False时返回str内容，否则返回AIMessage对象
        """
        app_logger.debug(f"调用模型: input={input}, kwargs={kwargs}")
        response: AIMessage = self._model.invoke(input, **kwargs)
        return response

    async def ainvoke(self, input: LanguageModelInput, **kwargs) -> Union[str, AIMessage]:
        """
        异步调用模型，符合ChatOpenAI接口规范
        Args:
            input: LanguageModelInput类型，支持str、List[BaseMessage]等格式
        Returns:
            tool_enabled=False时返回str内容，否则返回AIMessage对象
        """
        app_logger.debug(f"异步调用模型: input={input}, kwargs={kwargs}")
        response: AIMessage = await self._model.ainvoke(input, **kwargs)
        return response


class DeepSeekChatModel(BaseOpenAIChatModel):
    """DeepSeek聊天模型实现"""
    def __init__(self, tool_enabled: bool = False):
        super().__init__(
            api_key=settings.api.deepseek_api_key,
            base_url=settings.api.deepseek_api_base,
            model_name=settings.api.model,
            temperature=settings.api.temperature,
            max_tokens=settings.api.max_tokens,
            tool_enabled=tool_enabled
        )


# 保持原有类名兼容，方便现有代码过渡
LangChainChatAdapter = DeepSeekChatModel

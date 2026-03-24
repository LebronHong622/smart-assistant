"""通用问答生成节点工厂函数
处理normal意图，直接生成回答
"""
from typing import Dict, Any, List, Callable

from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage

from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.model_enums import ModelType, RoutingStrategy
from ..state import AgentState


def create_general_generate_node(
    prompt_port: PromptPort,
    model_router_port: ModelRouterPort,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建通用问答生成节点"""

    def _build_context(documents: List[Dict[str, Any]]) -> str:
        """构建文档上下文"""
        if not documents:
            return "无相关文档"
        return "\n\n".join(doc.get("content", "") for doc in documents)

    def generate_answer(state: AgentState) -> Dict[str, Any]:
        """通用问答生成：基于上下文（如果有）生成回答"""
        logger.info(f"[通用问答] 生成回答")
        llm = model_router_port.get_model(ModelType.CHAT, strategy=RoutingStrategy.DEFAULT)
        documents = state.get("relevant_documents", [])

        context = _build_context(documents)

        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.general_answer_prompt",
            query=state["query"],
            context=context,
            chat_history=state["chat_history"]
        )
        response: AIMessage = llm.invoke(prompt_value)
        answer = response.content.strip()

        logger.info(f"[通用问答] 回答生成完成，长度: {len(answer)}")
        result = {"answer": answer}
        if documents:
            result["documents"] = documents

        return result

    return generate_answer

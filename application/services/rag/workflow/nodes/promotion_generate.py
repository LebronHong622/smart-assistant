"""促销规则回答生成节点工厂函数
基于促销规则检索结果生成回答
"""
from typing import Dict, Any, List, Callable

from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage

from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.model_enums import ModelType, RoutingStrategy
from ..state import AgentState


def create_promotion_generate_node(
    prompt_port: PromptPort,
    model_router_port: ModelRouterPort,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建促销规则回答生成节点"""

    def _build_context(documents: List[Dict[str, Any]]) -> str:
        """构建上下文，每个文档格式化为 'promotion_name: content'"""
        if not documents:
            return "无相关文档"

        context_parts = []
        for doc in documents:
            content = doc.get("content", "")
            promotion_name = doc.get("metadata", {}).get("promotion_name", "")
            if promotion_name:
                # 按要求格式：promotion_name: content
                context_parts.append(f"{promotion_name}: {content}")
            else:
                context_parts.append(content)

        return "\n\n".join(context_parts)

    def generate_answer(state: AgentState) -> Dict[str, Any]:
        """基于促销规则文档生成回答"""
        logger.info(f"[促销规则] 生成回答，相关文档数量: {len(state['relevant_documents'])}")
        llm = model_router_port.get_model(ModelType.CHAT, strategy=RoutingStrategy.DEFAULT)
        documents = state["relevant_documents"]

        context = _build_context(documents)

        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.promotion_answer_prompt",
            query=state["query"],
            context=context,
            chat_history=state["chat_history"]
        )
        response: AIMessage = llm.invoke(prompt_value)
        answer = response.content.strip()

        logger.info(f"[促销规则] 回答生成完成，长度: {len(answer)}")
        return {
            "answer": answer,
            "documents": documents
        }

    return generate_answer

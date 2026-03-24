"""意图分类节点工厂函数
识别用户意图，映射到对应业务领域
"""
from typing import Dict, Any, Callable

from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage

from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.model_enums import ModelType, RoutingStrategy
from ..state import AgentState


def create_intent_classification_node(
    prompt_port: PromptPort,
    model_router_port: ModelRouterPort,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建意图分类节点工厂函数"""

    def classify_intent(state: AgentState) -> Dict[str, Any]:
        """意图分类：识别用户问题属于哪个业务领域"""
        logger.info(f"意图分类: query={state['query']}")
        llm = model_router_port.get_model(ModelType.CHAT, strategy=RoutingStrategy.DEFAULT)

        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.intent_classification_prompt",
            query=state["query"],
            chat_history=state["chat_history"]
        )
        response: AIMessage = llm.invoke(prompt_value)
        intent = response.content.strip()

        # 中文分类结果映射到内部domain名称
        intent_mapping = {
            "商品导购": "product_selling_points",
            "售后规则": "after_sales_policy",
            "促销规则": "promotion_rules",
            "normal": "normal",
        }
        domain = intent_mapping.get(intent, "normal")
        needs_retrieval = domain != "normal"

        logger.info(f"意图分类结果: raw={intent}, mapped={domain}, needs_retrieval={needs_retrieval}")

        return {
            "intent": domain,
            "needs_retrieval": needs_retrieval,
            "rewrite_count": 0,
        }

    return classify_intent

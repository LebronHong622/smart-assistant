"""LangGraph workflow builder
三级RAG工作流：意图识别 → 领域检索 → 生成
"""
from typing import Any, Callable
from langgraph.graph import StateGraph, END

from domain.document.service.rag_processing_service import RAGProcessingServiceFactory
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from .state import AgentState
from .nodes import (
    create_intent_classification_node,
    create_product_retrieve_node,
    create_after_sales_retrieve_node,
    create_promotion_retrieve_node,
    create_generate_node
)


def build_rag_workflow(
    prompt_port: PromptPort,
    model_router_port: ModelRouterPort,
    logger: LoggerPort,
    rag_processing_service_factory: RAGProcessingServiceFactory,
    document_repository_factory: Callable[[], DocumentRepository],
    default_retrieve_limit: int = 5,
) -> StateGraph:
    """
    构建三级RAG工作流：意图识别 → 领域检索 → 生成

    Args:
        prompt_port: Prompt服务端口
        model_router_port: 模型路由端口，用于获取LLM实例
        logger: 日志服务
        rag_processing_service_factory: RAG处理服务工厂
        document_repository_factory: DocumentRepository 工厂函数，每个领域创建独立实例
        default_retrieve_limit: 默认检索返回数量

    Returns:
        StateGraph 工作流实例
    """
    # 在内部创建所有节点（每个检索节点内部自己初始化rag_service）
    intent_node = create_intent_classification_node(prompt_port, model_router_port, logger)
    product_retrieve_node = create_product_retrieve_node(
        rag_processing_service_factory, document_repository_factory, logger, default_retrieve_limit
    )
    after_sales_retrieve_node = create_after_sales_retrieve_node(
        rag_processing_service_factory, document_repository_factory, logger, default_retrieve_limit
    )
    promotion_retrieve_node = create_promotion_retrieve_node(
        rag_processing_service_factory, document_repository_factory, logger, default_retrieve_limit
    )
    generate_node = create_generate_node(prompt_port, model_router_port, logger)

    # 构建图
    workflow = StateGraph(AgentState)

    # 添加所有节点
    workflow.add_node("intent_classification", intent_node)
    workflow.add_node("retrieve_product", product_retrieve_node)
    workflow.add_node("retrieve_after_sales", after_sales_retrieve_node)
    workflow.add_node("retrieve_promotion", promotion_retrieve_node)
    workflow.add_node("generate_answer", generate_node)

    # 设置入口
    workflow.set_entry_point("intent_classification")

    # 意图分类后的条件路由
    def route_by_intent(state: AgentState) -> str:
        """根据意图路由到对应检索节点"""
        intent = state["intent"]
        routing_map = {
            "product_selling_points": "retrieve_product",
            "after_sales_policy": "retrieve_after_sales",
            "promotion_rules": "retrieve_promotion",
            "normal": "generate_answer",
        }
        route_to = routing_map.get(intent, "generate_answer")
        logger.debug(f"意图路由: intent={intent}, route_to={route_to}")
        return route_to

    workflow.add_conditional_edges(
        "intent_classification",
        route_by_intent
    )

    # 所有检索节点完成后都跳转到生成
    workflow.add_edge("retrieve_product", "generate_answer")
    workflow.add_edge("retrieve_after_sales", "generate_answer")
    workflow.add_edge("retrieve_promotion", "generate_answer")

    # 生成就是终点
    workflow.add_edge("generate_answer", END)

    return workflow

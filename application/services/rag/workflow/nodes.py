"""Workflow node factory functions
三级RAG流程：意图分类 → 领域检索 → 生成
"""
from typing import Dict, Any, List, Callable
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage

from domain.document.service.rag_processing_service import RAGProcessingServiceFactory
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.model_enums import ModelType, RoutingStrategy
from .state import AgentState


def create_intent_classification_node(
    prompt_port: PromptPort,
    model_router_port: ModelRouterPort,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建意图分类节点工厂函数
    识别用户意图，映射到对应业务领域
    """

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


def create_product_retrieve_node(
    rag_processing_service_factory: RAGProcessingServiceFactory,
    document_repository_factory: Callable[[], DocumentRepository],
    logger: LoggerPort,
    default_limit: int = 5
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建商品导购检索节点，内部初始化对应领域的RAG服务"""
    # 在工厂创建时就初始化好rag_service，闭包持有
    doc_repo = document_repository_factory()
    rag_service = rag_processing_service_factory.create_service(
        domain="product_selling_points",
        document_repository=doc_repo
    )

    def retrieve_documents(state: AgentState) -> Dict[str, Any]:
        query = state["rewritten_query"] or state["query"]
        logger.info(f"执行商品导购检索: query={query[:50]}...")
        domain_documents = rag_service.retrieve_similar(
            query=query, limit=default_limit, score_threshold=0.7
        )
        documents = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": doc.metadata or {},
                "similarity_score": doc.similarity_score,
                "domain": "product_selling_points"
            }
            for doc in domain_documents
        ]
        logger.info(f"商品导购检索完成，返回 {len(documents)} 个文档")
        return {"retrieved_documents": documents, "relevant_documents": documents}

    return retrieve_documents


def create_after_sales_retrieve_node(
    rag_processing_service_factory: RAGProcessingServiceFactory,
    document_repository_factory: Callable[[], DocumentRepository],
    logger: LoggerPort,
    default_limit: int = 5
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建售后规则检索节点，内部初始化对应领域的RAG服务"""
    doc_repo = document_repository_factory()
    rag_service = rag_processing_service_factory.create_service(
        domain="after_sales_policy",
        document_repository=doc_repo
    )

    def retrieve_documents(state: AgentState) -> Dict[str, Any]:
        query = state["rewritten_query"] or state["query"]
        logger.info(f"执行售后规则检索: query={query[:50]}...")
        domain_documents = rag_service.retrieve_similar(
            query=query, limit=default_limit, score_threshold=0.7
        )
        documents = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": doc.metadata or {},
                "similarity_score": doc.similarity_score,
                "domain": "after_sales_policy"
            }
            for doc in domain_documents
        ]
        logger.info(f"售后规则检索完成，返回 {len(documents)} 个文档")
        return {"retrieved_documents": documents, "relevant_documents": documents}

    return retrieve_documents


def create_promotion_retrieve_node(
    rag_processing_service_factory: RAGProcessingServiceFactory,
    document_repository_factory: Callable[[], DocumentRepository],
    logger: LoggerPort,
    default_limit: int = 5
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建促销规则检索节点，内部初始化对应领域的RAG服务"""
    doc_repo = document_repository_factory()
    rag_service = rag_processing_service_factory.create_service(
        domain="promotion_rules",
        document_repository=doc_repo
    )

    def retrieve_documents(state: AgentState) -> Dict[str, Any]:
        query = state["rewritten_query"] or state["query"]
        logger.info(f"执行促销规则检索: query={query[:50]}...")
        domain_documents = rag_service.retrieve_similar(
            query=query, limit=default_limit, score_threshold=0.7
        )
        documents = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": doc.metadata or {},
                "similarity_score": doc.similarity_score,
                "domain": "promotion_rules"
            }
            for doc in domain_documents
        ]
        logger.info(f"促销规则检索完成，返回 {len(documents)} 个文档")
        return {"retrieved_documents": documents, "relevant_documents": documents}

    return retrieve_documents


def create_generate_node(
    prompt_port: PromptPort,
    model_router_port: ModelRouterPort,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建回答生成节点工厂函数"""

    def _build_context(documents: List[Dict[str, Any]]) -> str:
        """构建文档上下文"""
        if not documents:
            return "无相关文档"
        return "\n\n".join(doc.get("content", "") for doc in documents)

    def generate_answer(state: AgentState) -> Dict[str, Any]:
        """回答生成节点：基于相关文档生成最终回答"""
        logger.info(f"生成回答，相关文档数量: {len(state['relevant_documents'])}")
        llm = model_router_port.get_model(ModelType.CHAT, strategy=RoutingStrategy.DEFAULT)

        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.answer_prompt",
            query=state["query"],
            context=_build_context(state["relevant_documents"]),
            chat_history=state["chat_history"]
        )
        response: AIMessage = llm.invoke(prompt_value)
        answer = response.content.strip()

        logger.info(f"回答生成完成，长度: {len(answer)}")
        return {"answer": answer}

    return generate_answer

"""售后规则检索节点工厂函数
执行售后规则知识库检索
"""
from typing import Dict, Any, List, Callable

from domain.document.service.rag_processing_service import RAGProcessingServiceFactory
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.logger_port import LoggerPort
from ..state import AgentState


def create_after_sales_retrieve_node(
    rag_processing_service_factory: RAGProcessingServiceFactory,
    document_repository_factory: Callable[[], DocumentRepository],
    logger: LoggerPort,
    default_limit: int = 5
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建售后规则检索节点，执行时初始化对应领域的RAG服务"""

    def retrieve_documents(state: AgentState) -> Dict[str, Any]:
        # 执行时初始化 rag_service（单例模式，重复调用返回同一实例）
        doc_repo = document_repository_factory()
        rag_service = rag_processing_service_factory.create_service(
            domain="after_sales_policy",
            document_repository=doc_repo
        )

        query = state["rewritten_query"] or state["query"]
        logger.info(f"执行售后规则检索: query={query[:50]}...")
        domain_documents = rag_service.retrieve_similar(
            query=query, limit=default_limit, score_threshold=0.7
        )
        documents = [
            {
                "id": str(doc.id),
                "content": doc.content,
                "metadata": {
                    "policy_id": doc.metadata.get("policy_id"),
                    "policy_name": doc.metadata.get("policy_name"),
                    **(doc.metadata or {}),
                },
                "similarity_score": doc.similarity_score,
                "domain": "after_sales_policy"
            }
            for doc in domain_documents
        ]
        logger.info(f"售后规则检索完成，返回 {len(documents)} 个文档")
        return {"retrieved_documents": documents, "relevant_documents": documents}

    return retrieve_documents

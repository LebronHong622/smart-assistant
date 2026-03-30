"""
Agentic RAG Service Implementation
基于 LangGraph 的三级RAG智能检索工作流服务
意图识别 → 领域检索 → 生成
"""
from typing import List, Dict, Any, Optional, Callable
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage
from langchain.tools import BaseTool

from application.services.document.rag_processing_service import RAGProcessingServiceFactory
from domain.repository.document.document_repository import DocumentRepository
from domain.vo.conversation.rag_conversation_state import RAGConversationState
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.tool_port import ToolPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.model_enums import ModelType

from .agentic_rag_workflow import AgenticRAGWorkflow
from .workflow import AgentState, build_rag_workflow
from .mappers import StateMapper


class LangchainAgenticRagServiceImpl(AgenticRAGWorkflow):
    """
    Agentic RAG 服务实现（三级RAG版本）

    使用 LangGraph 实现完整的三级检索工作流：
    - 意图分类（商品导购/售后规则/促销规则/normal）
    - 根据意图路由到对应领域检索
    - 基于检索结果生成回答
    """

    def __init__(
        self,
        logger: LoggerPort,
        tool_port: ToolPort,
        prompt_port: PromptPort,
        model_router_port: ModelRouterPort,
        # 注入工厂，构造函数简洁，不需要列出三个服务
        rag_processing_service_factory: RAGProcessingServiceFactory,
        document_repository_factory: Callable[[], DocumentRepository],
        default_retrieve_limit: int = 5,
    ):
        self.logger = logger
        self.tool_port = tool_port
        self.prompt_port = prompt_port
        self.model_router_port = model_router_port
        self.llm = model_router_port.get_model(ModelType.CHAT)
        self.tools: List[BaseTool] = self.tool_port.get_tools(agent_type="agentic_rag")

        # 构建工作流（build_rag_workflow 内部创建所有节点和初始化服务）
        self.workflow = build_rag_workflow(
            prompt_port=self.prompt_port,
            model_router_port=self.model_router_port,
            logger=self.logger,
            rag_processing_service_factory=rag_processing_service_factory,
            document_repository_factory=document_repository_factory,
            default_retrieve_limit=default_retrieve_limit
        )
        self.app = self.workflow.compile()
        self.logger.info("三级RAG工作流初始化完成")

    def _call_llm(self, prompt_key: str, **format_kwargs) -> str:
        """通用 LLM 调用方法"""
        prompt_value: PromptValue = self.prompt_port.get_prompt(prompt_key, **format_kwargs)
        response: AIMessage = self.llm.invoke(prompt_value)
        return response.content.strip()

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建文档上下文"""
        if not documents:
            return "无相关文档"
        return "\n\n".join(doc.get("content", "") for doc in documents)

    def _grade_single_document(self, query: str, doc: Dict[str, Any]) -> bool:
        """评估单个文档是否相关"""
        content = self._call_llm(
            "agentic_rag.grade_prompt",
            query=query,
            document_content=doc.get("content", "")
        ).lower()
        return "relevant" in content or "yes" in content

    # ========== 接口实现 ==========

    def execute_workflow(
        self,
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict]] = None
    ) -> RAGConversationState:
        """执行完整工作流"""
        self.logger.info(f"执行 Agentic RAG 工作流，session_id={session_id}, query={query}")
        chat_history = chat_history or []

        try:
            initial_state = StateMapper.to_agent_state(
                query=query,
                session_id=session_id,
                chat_history=chat_history
            )

            result = self.app.invoke(initial_state)
            self.logger.info(f"工作流执行完成，session_id={session_id}")

            return StateMapper.to_rag_conversation_state(
                result=result,
                session_id=session_id,
                query=query,
                chat_history=chat_history
            )

        except Exception as e:
            self.logger.error(f"工作流执行失败: {str(e)}", exc_info=True)
            return StateMapper.to_error_rag_conversation_state(
                session_id=session_id,
                query=query,
                chat_history=chat_history,
                error=str(e)
            )

    def grade_documents_relevance(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """评估文档相关性"""
        return [doc for doc in documents if self._grade_single_document(query, doc)]

    def rewrite_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """重写查询"""
        return self._call_llm(
            "agentic_rag.rewrite_prompt",
            original_query=query,
            chat_history=chat_history or [],
            rewrite_count=1
        )

    def generate_answer(
        self,
        query: str,
        relevant_documents: List[Dict[str, Any]],
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """生成回答"""
        return self._call_llm(
            "agentic_rag.answer_prompt",
            query=query,
            context=self._build_context(relevant_documents),
            chat_history=chat_history or []
        )

    def should_retrieve(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> bool:
        """判断是否需要检索"""
        return "retrieve" in self._call_llm(
            "agentic_rag.route_prompt",
            query=query,
            chat_history=chat_history or []
        ).lower()

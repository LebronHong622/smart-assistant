"""
Agentic RAG Service Implementation
基于 LangGraph 的智能检索工作流服务
"""
from typing import List, Dict, Any, Optional
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage
from langchain.tools import BaseTool

from domain.qa.service.agentic_rag_service import AgenticRagService
from domain.qa.value_object.rag_state import RagState
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.tool_port import ToolPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.model_enums import ModelType

from .workflow import AgentState, build_rag_workflow
from .mappers import StateMapper


class LangchainAgenticRagServiceImpl(AgenticRagService):
    """
    Agentic RAG 服务实现
    
    使用 LangGraph 实现完整的智能检索工作流，包括：
    - 查询路由（判断是否需要检索）
    - 文档检索
    - 文档相关性评估
    - 查询重写
    - 回答生成
    """

    def __init__(
        self,
        logger: LoggerPort,
        tool_port: ToolPort,
        prompt_port: PromptPort,
        model_router_port: ModelRouterPort,
        max_rewrite_attempts: int = 2
    ):
        self.logger = logger
        self.tool_port = tool_port
        self.prompt_port = prompt_port
        self.llm = model_router_port.get_model(ModelType.CHAT)
        self.tools: List[BaseTool] = self.tool_port.get_tools(agent_type="agentic_rag")
        self.max_rewrite_attempts = max_rewrite_attempts
        
        # 构建并编译工作流
        self.workflow = build_rag_workflow(
            prompt_port=self.prompt_port,
            llm=self.llm,
            tools=self.tools,
            logger=self.logger,
            max_rewrite_attempts=self.max_rewrite_attempts
        )
        self.app = self.workflow.compile()

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
    ) -> RagState:
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

            return StateMapper.to_rag_state(
                result=result,
                session_id=session_id,
                query=query,
                chat_history=chat_history
            )

        except Exception as e:
            self.logger.error(f"工作流执行失败: {str(e)}", exc_info=True)
            return StateMapper.to_error_rag_state(
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

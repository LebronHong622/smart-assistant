from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompt_values import PromptValue
from langchain.tools import BaseTool

from domain.qa.service.agentic_rag_service import AgenticRagService
from domain.qa.value_object.rag_state import RagState
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.tool_port import ToolPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.ports.model_capability_port import BaseModel
from domain.shared.model_enums import ModelType


class AgentState(TypedDict):
    """LangGraph 工作流状态定义"""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    rewritten_query: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    needs_retrieval: bool
    rewrite_count: int
    answer: Optional[str]
    session_id: str
    chat_history: List[Dict]


class LangchainAgenticRagServiceImpl(AgenticRagService):
    """
    Agentic RAG 服务实现
    基于 LangGraph 实现完整的智能检索工作流
    """

    def __init__(
        self,
        logger: LoggerPort,
        tool_port: ToolPort,
        prompt_port: PromptPort,
        model_router_port: ModelRouterPort,
        max_rewrite_attempts: int = 2,
        max_retrieval_attempts: int = 2
    ):
        self.logger = logger
        self.tool_port = tool_port
        self.prompt_port = prompt_port
        self.llm = model_router_port.get_model(ModelType.CHAT)
        self.tools: List[BaseTool] = self.tool_port.get_tools(agent_type="agentic_rag")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        # 配置参数
        self.max_rewrite_attempts = max_rewrite_attempts
        self.max_retrieval_attempts = max_retrieval_attempts

    # ========== 通用方法 ==========

    def _call_llm(self, prompt_key: str, **format_kwargs) -> str:
        """
        通用 LLM 调用方法
        """
        message: PromptValue = self.prompt_port.get_prompt(prompt_key, **format_kwargs)
        response: AIMessage = self.llm.invoke(message)
        return response.content.strip()

    def _get_query(self, state: AgentState) -> str:
        """获取当前查询（优先使用重写后的查询）"""
        return state["rewritten_query"] or state["query"]

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

    # ========== 工作流构建 ==========

    def _build_workflow(self) -> StateGraph:
        """
        构建 LangGraph 工作流
        """
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("route", self._route_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("generate_answer", self._generate_answer)

        # 设置入口
        workflow.set_entry_point("route")

        # 配置边
        workflow.add_conditional_edges(
            "route",
            lambda x: "retrieve" if x["needs_retrieval"] else "generate_answer"
        )
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_next_step_after_grading
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate_answer", END)

        return workflow

    # ========== 工作流节点 ==========

    def _route_query(self, state: AgentState) -> Dict[str, Any]:
        """路由节点：判断是否需要检索或直接回答"""
        self.logger.info(f"路由决策: query={state['query']}")

        content = self._call_llm(
            "agentic_rag.route_prompt",
            query=state["query"],
            chat_history=state["chat_history"]
        ).lower()

        needs_retrieval = "retrieve" in content
        self.logger.info(f"路由决策结果: {'需要检索' if needs_retrieval else '直接回答'}")

        return {**state, "needs_retrieval": needs_retrieval}

    def _retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        """检索节点：调用文档检索工具"""
        query = self._get_query(state)
        self.logger.info(f"执行文档检索: query={query}")

        try:
            retrieval_tool = next(tool for tool in self.tools if tool.name == "langchain_document_retrieval")
            documents = retrieval_tool.invoke({"query": query, "limit": 5})

            self.logger.info(f"检索完成，返回 {len(documents)} 个文档")
            return {**state, "retrieved_documents": documents, "rewritten_query": query}
        except Exception as e:
            self.logger.error(f"文档检索失败: {str(e)}")
            return {**state, "retrieved_documents": []}

    def _grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """文档评估节点：评估文档与查询的相关性"""
        query = self._get_query(state)
        documents = state["retrieved_documents"]
        self.logger.info(f"评估文档相关性，共 {len(documents)} 个文档")

        if not documents:
            return {**state, "relevant_documents": []}

        relevant_docs = [
            doc for doc in documents
            if self._grade_single_document(query, doc)
        ]

        self.logger.info(f"评估完成，共 {len(relevant_docs)} 个相关文档")
        return {**state, "relevant_documents": relevant_docs}

    def _decide_next_step_after_grading(self, state: AgentState) -> str:
        """评估后决策：判断下一步操作"""
        relevant_docs = state["relevant_documents"]
        rewrite_count = state["rewrite_count"]

        if relevant_docs:
            return "generate_answer"
        elif rewrite_count < self.max_rewrite_attempts:
            return "rewrite_query"
        else:
            self.logger.warning(f"已达到最大重写次数 {self.max_rewrite_attempts}，直接生成回答")
            return "generate_answer"

    def _rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """查询重写节点：优化查询以提升检索效果"""
        rewrite_count = state["rewrite_count"] + 1
        self.logger.info(f"第 {rewrite_count} 次重写查询: {state['query']}")

        rewritten_query = self._call_llm(
            "agentic_rag.rewrite_prompt",
            original_query=state["query"],
            chat_history=state["chat_history"],
            rewrite_count=rewrite_count
        )

        self.logger.info(f"重写后查询: {rewritten_query}")
        return {**state, "rewritten_query": rewritten_query, "rewrite_count": rewrite_count}

    def _generate_answer(self, state: AgentState) -> Dict[str, Any]:
        """回答生成节点：基于相关文档生成最终回答"""
        self.logger.info(f"生成回答，相关文档数量: {len(state['relevant_documents'])}")

        answer = self._call_llm(
            "agentic_rag.answer_prompt",
            query=state["query"],
            context=self._build_context(state["relevant_documents"]),
            chat_history=state["chat_history"]
        )

        self.logger.info(f"回答生成完成，长度: {len(answer)}")
        return {**state, "answer": answer}

    # ========== 接口实现 ==========

    def execute_workflow(self, query: str, session_id: str, chat_history: Optional[List[Dict]] = None) -> RagState:
        """执行完整工作流"""
        self.logger.info(f"执行 Agentic RAG 工作流，session_id={session_id}, query={query}")
        chat_history = chat_history or []

        try:
            initial_state = {
                "query": query,
                "rewritten_query": None,
                "retrieved_documents": [],
                "relevant_documents": [],
                "needs_retrieval": False,
                "rewrite_count": 0,
                "answer": None,
                "session_id": session_id,
                "chat_history": chat_history,
                "messages": []
            }

            result = self.app.invoke(initial_state)
            self.logger.info(f"工作流执行完成，session_id={session_id}")

            rag_state = RagState(
                session_id=session_id,
                query=query,
                chat_history=chat_history,
                retrieved_documents=result.get("retrieved_documents", []),
                relevant_documents=result.get("relevant_documents", []),
                rewritten_query=result.get("rewritten_query"),
                rewrite_count=result.get("rewrite_count", 0),
                answer=result.get("answer"),
                needs_retrieval=result.get("needs_retrieval", False)
            )

            return rag_state

        except Exception as e:
            self.logger.error(f"工作流执行失败: {str(e)}", exc_info=True)
            return RagState(
                session_id=session_id,
                query=query,
                chat_history=chat_history,
                error=str(e)
            )

    def grade_documents_relevance(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """评估文档相关性（接口实现）"""
        return [doc for doc in documents if self._grade_single_document(query, doc)]

    def rewrite_query(self, query: str, chat_history: Optional[List[Dict]] = None) -> str:
        """重写查询（接口实现）"""
        return self._call_llm(
            "agentic_rag.rewrite_prompt",
            original_query=query,
            chat_history=chat_history or [],
            rewrite_count=1
        )

    def generate_answer(self, query: str, relevant_documents: List[Dict[str, Any]],
                        chat_history: Optional[List[Dict]] = None) -> str:
        """生成回答（接口实现）"""
        return self._call_llm(
            "agentic_rag.answer_prompt",
            query=query,
            context=self._build_context(relevant_documents),
            chat_history=chat_history or []
        )

    def should_retrieve(self, query: str, chat_history: Optional[List[Dict]] = None) -> bool:
        """判断是否需要检索（接口实现）"""
        return "retrieve" in self._call_llm(
            "agentic_rag.route_prompt",
            query=query,
            chat_history=chat_history or []
        ).lower()

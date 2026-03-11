from typing import List, Dict, Optional
from uuid import uuid4

from domain.qa.value_object.rag_state import RagState
from domain.qa.service.agentic_rag_service import AgenticRagService
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.memory_port import MemoryPort


class AgenticRagAgent:
    """
    Agentic RAG 代理
    完全独立的代理实现，与现有QA代理隔离
    """

    def __init__(
        self,
        rag_service: AgenticRagService,
        memory_port: MemoryPort,
        logger: LoggerPort,
        session_id: Optional[str] = None
    ):
        self.session_id = session_id or str(uuid4())
        self.rag_service = rag_service
        self.memory_port = memory_port
        self.logger = logger
        self.logger.info(f"Agentic RAG 代理初始化完成，session_id={self.session_id}")

    def chat(self, query: str) -> str:
        """
        发送消息并获取回答
        Args:
            query: 用户查询
        Returns:
            回答内容
        """
        self.logger.info(f"收到用户查询，session_id={self.session_id}, query={query}")

        try:
            # 获取历史对话
            chat_history = self.memory_port.get_history(self.session_id)

            # 执行RAG工作流
            rag_state = self.rag_service.execute_workflow(
                query=query,
                session_id=self.session_id,
                chat_history=chat_history
            )

            if rag_state.error:
                self.logger.error(f"工作流执行错误: {rag_state.error}")
                return f"抱歉，处理您的请求时出现错误：{rag_state.error}"

            # 保存对话历史
            self.memory_port.add_user_message(self.session_id, query)
            if rag_state.answer:
                self.memory_port.add_assistant_message(self.session_id, rag_state.answer)

            self.logger.info(f"返回回答，session_id={self.session_id}, answer_length={len(rag_state.answer)}")
            return rag_state.answer or "抱歉，我无法回答您的问题。"

        except Exception as e:
            self.logger.error(f"聊天处理失败: {str(e)}", exc_info=True)
            return "抱歉，处理您的请求时出现未知错误。"

    def get_session_history(self) -> List[Dict]:
        """
        获取会话历史
        """
        return self.memory_port.get_history(self.session_id)

    def clear_session(self) -> None:
        """
        清空会话
        """
        self.memory_port.clear_history(self.session_id)
        self.logger.info(f"会话已清空，session_id={self.session_id}")

"""
问答智能代理模块
基于 LangChain create_agent 函数构建的智能问答助手
"""

from typing import Optional
from datetime import datetime
import uuid

from domain.qa.entity.qa_conversation import QAConversation
from domain.qa.value_object.qa_query import QAQuery
from domain.qa.value_object.qa_response import QAResponse
from domain.qa.service.qa_service import QAService
from infrastructure.log import app_logger

class QAAgent:
    """问答代理"""
    def __init__(self, session_id: str):
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation = QAConversation.create(self.session_id)
        self.qa_service = QAService()

    def chat(self, query: str) -> dict:
        """处理用户查询"""
        qa_query = QAQuery(content=query)
        qa_response = self.qa_service.process_query(self.conversation, qa_query)

        self.conversation.add_query(qa_query)
        self.conversation.add_response(qa_response)

        return {
            "message": qa_response.content,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }

def create_qa_agent(session_id: Optional[str] = None) -> QAAgent:
    """创建QA代理实例"""
    return QAAgent(session_id=session_id)

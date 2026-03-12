"""AgentState definition for LangGraph workflow"""
from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """LangGraph 工作流状态定义
    
    Attributes:
        messages: 消息列表，支持累加
        query: 原始用户查询
        rewritten_query: 重写后的查询（可选）
        retrieved_documents: 检索到的文档列表
        relevant_documents: 相关文档列表
        needs_retrieval: 是否需要检索
        rewrite_count: 查询重写次数
        answer: 生成的回答
        session_id: 会话ID
        chat_history: 对话历史
    """
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

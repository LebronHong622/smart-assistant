"""
问答智能代理模块
基于 LangChain create_agent 函数构建的智能问答助手
"""

from typing import Optional
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import uuid

from tools import tool_manager
from models import model_manager
from config.log import app_logger
from memory import MemoryManager
from config.settings import settings

class QAAgent:
    def __init__(self, session_id: str):
        # 使用sessionid管理会话
        self.session_id = session_id or str(uuid.uuid4())

        # 初始化工具
        self.tools = tool_manager.init_tools()

        # 初始化大模型
        self.model = model_manager.get_default_model()

        self.memory_manager = MemoryManager()

        self.middleware = self._get_middleware()
        
        # 使用 create_agent 创建代理
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """使用 create_agent 创建智能代理"""
        # 创建代理
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt="你是一个专业的问答助手，能够根据用户问题调用工具获取答案。",
            checkpointer=self.memory_manager.get_in_memory(),
            middleware=self.middleware
        )
        
        return agent

    def _get_middleware(self) -> list:
        """获取内存管理中间件"""
        middle_ware = []
        middle_ware.append(self.memory_manager.get_overflow_memory_middleware("summary"))
        return middle_ware
    
    def chat(self, query: str) -> str:
        """处理用户查询"""
        try:
            # 使用代理处理用户查询
            query_message = HumanMessage(content=query)
            prompt_message = {
                "messages": [query_message]
            }

            response = self.agent.invoke(
                prompt_message, 
                self.memory_manager.get_thread_memory_config(self.session_id), 
            )
            
            # 返回代理的输出
            return response["messages"][-1].content
        except Exception as e:
            app_logger.error(f"对话处理失败: {e}")
            return f"处理查询时出错: {str(e)}"
    
def create_qa_agent(session_id: Optional[str] = None) -> QAAgent:
    """创建QA代理实例"""
    return QAAgent(session_id=session_id)

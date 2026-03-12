"""LangGraph workflow builder"""
from typing import List, Any
from langgraph.graph import StateGraph, END
from langchain.tools import BaseTool

from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from .state import AgentState
from .nodes import (
    create_route_node,
    create_retrieve_node,
    create_grade_node,
    create_rewrite_node,
    create_generate_node
)


def build_rag_workflow(
    prompt_port: PromptPort,
    llm: Any,
    tools: List[BaseTool],
    logger: LoggerPort,
    max_rewrite_attempts: int = 2
) -> StateGraph:
    """
    构建 Agentic RAG 工作流
    
    Args:
        prompt_port: Prompt服务端口
        llm: 语言模型实例
        tools: 工具列表
        logger: 日志服务
        max_rewrite_attempts: 最大查询重写次数
        
    Returns:
        编译后的 StateGraph
    """
    # 创建节点
    route_node = create_route_node(prompt_port, llm, logger)
    retrieve_node = create_retrieve_node(tools, logger)
    grade_node = create_grade_node(prompt_port, llm, logger)
    rewrite_node = create_rewrite_node(prompt_port, llm, logger)
    generate_node = create_generate_node(prompt_port, llm, logger)
    
    # 构建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_node)
    workflow.add_node("rewrite_query", rewrite_node)
    workflow.add_node("generate_answer", generate_node)
    
    # 设置入口
    workflow.set_entry_point("route")
    
    # 配置边
    workflow.add_conditional_edges(
        "route",
        lambda state: "retrieve" if state["needs_retrieval"] else "generate_answer"
    )
    workflow.add_edge("retrieve", "grade_documents")
    
    def decide_after_grading(state: AgentState) -> str:
        """评估后决策"""
        relevant_docs = state["relevant_documents"]
        rewrite_count = state["rewrite_count"]
        
        if relevant_docs:
            return "generate_answer"
        elif rewrite_count < max_rewrite_attempts:
            return "rewrite_query"
        else:
            logger.warning(f"已达到最大重写次数 {max_rewrite_attempts}，直接生成回答")
            return "generate_answer"
    
    workflow.add_conditional_edges("grade_documents", decide_after_grading)
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)
    
    return workflow

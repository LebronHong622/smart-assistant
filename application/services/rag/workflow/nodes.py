"""Workflow node factory functions"""
from typing import Dict, Any, List, Callable
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import AIMessage
from langchain.tools import BaseTool

from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from .state import AgentState


def create_route_node(
    prompt_port: PromptPort,
    llm: Any,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建路由节点工厂函数"""
    
    def route_query(state: AgentState) -> Dict[str, Any]:
        """路由节点：判断是否需要检索或直接回答"""
        logger.info(f"路由决策: query={state['query']}")
        
        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.route_prompt",
            query=state["query"],
            chat_history=state["chat_history"]
        )
        response: AIMessage = llm.invoke(prompt_value)
        content = response.content.strip().lower()
        
        needs_retrieval = "retrieve" in content
        logger.info(f"路由决策结果: {'需要检索' if needs_retrieval else '直接回答'}")
        
        return {"needs_retrieval": needs_retrieval}
    
    return route_query


def create_retrieve_node(
    tools: List[BaseTool],
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建检索节点工厂函数"""
    
    def retrieve_documents(state: AgentState) -> Dict[str, Any]:
        """检索节点：调用文档检索工具"""
        query = state["rewritten_query"] or state["query"]
        logger.info(f"执行文档检索: query={query}")
        
        try:
            retrieval_tool = next(
                (tool for tool in tools if tool.name == "langchain_document_retrieval"),
                None
            )
            if not retrieval_tool:
                logger.error("未找到文档检索工具")
                return {"retrieved_documents": [], "rewritten_query": query}
            
            documents = retrieval_tool.invoke({"query": query, "limit": 5})
            logger.info(f"检索完成，返回 {len(documents)} 个文档")
            return {"retrieved_documents": documents, "rewritten_query": query}
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return {"retrieved_documents": []}
    
    return retrieve_documents


def create_grade_node(
    prompt_port: PromptPort,
    llm: Any,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建文档评估节点工厂函数"""
    
    def _grade_single_document(query: str, doc: Dict[str, Any]) -> bool:
        """评估单个文档是否相关"""
        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.grade_prompt",
            query=query,
            document_content=doc.get("content", "")
        )
        response: AIMessage = llm.invoke(prompt_value)
        content = response.content.strip().lower()
        return "relevant" in content or "yes" in content
    
    def grade_documents(state: AgentState) -> Dict[str, Any]:
        """文档评估节点：评估文档与查询的相关性"""
        query = state["rewritten_query"] or state["query"]
        documents = state["retrieved_documents"]
        logger.info(f"评估文档相关性，共 {len(documents)} 个文档")
        
        if not documents:
            return {"relevant_documents": []}
        
        relevant_docs = [
            doc for doc in documents
            if _grade_single_document(query, doc)
        ]
        
        logger.info(f"评估完成，共 {len(relevant_docs)} 个相关文档")
        return {"relevant_documents": relevant_docs}
    
    return grade_documents


def create_rewrite_node(
    prompt_port: PromptPort,
    llm: Any,
    logger: LoggerPort
) -> Callable[[AgentState], Dict[str, Any]]:
    """创建查询重写节点工厂函数"""
    
    def rewrite_query(state: AgentState) -> Dict[str, Any]:
        """查询重写节点：优化查询以提升检索效果"""
        rewrite_count = state["rewrite_count"] + 1
        logger.info(f"第 {rewrite_count} 次重写查询: {state['query']}")
        
        prompt_value: PromptValue = prompt_port.get_prompt(
            "agentic_rag.rewrite_prompt",
            original_query=state["query"],
            chat_history=state["chat_history"],
            rewrite_count=rewrite_count
        )
        response: AIMessage = llm.invoke(prompt_value)
        rewritten_query = response.content.strip()
        
        logger.info(f"重写后查询: {rewritten_query}")
        return {"rewritten_query": rewritten_query, "rewrite_count": rewrite_count}
    
    return rewrite_query


def create_generate_node(
    prompt_port: PromptPort,
    llm: Any,
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

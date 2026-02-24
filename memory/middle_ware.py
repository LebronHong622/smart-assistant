"""
记忆中间件模块
"""
from langchain.agents.middleware import before_model, after_model, SummarizationMiddleware
from langchain.agents import AgentState
from config.settings import settings
from langchain.messages import RemoveMessage
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing import Any

MAX_MESSAGES_TO_KEEP = settings.app.max_session_history



@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """在模型调用前执行的中间件"""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES_TO_KEEP:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-MAX_MESSAGES_TO_KEEP:] if len(messages) % 2 == 0 else messages[-MAX_MESSAGES_TO_KEEP-1:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > MAX_MESSAGES_TO_KEEP:
        # remove the earliest messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-MAX_MESSAGES_TO_KEEP]]}
    return None


def summarize_messages() -> SummarizationMiddleware:
    """Summarize messages to keep conversation manageable."""
    return SummarizationMiddleware(
            model=settings.api.model,
            max_tokens_before_summary=settings.app.max_tokens_before_summary,  # Trigger summarization at 4000 tokens
            messages_to_keep=MAX_MESSAGES_TO_KEEP,  # Keep last 20 messages after summary
    )
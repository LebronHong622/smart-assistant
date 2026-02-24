# 内存管理类
from langgraph.checkpoint.memory import InMemorySaver
from memory.middle_ware import trim_messages, summarize_messages, delete_old_messages
from enums.enums import OverflowMemoryMethod

# 内存管理类
class MemoryManager:
    """Manages in-memory conversation checkpoints for LangGraph workflows."""
    def __init__(self):
        self.in_memory = InMemorySaver()
    
    def get_in_memory(self):
        """Return the InMemorySaver instance."""
        return self.in_memory
  
    def get_thread_memory_config(self, thread_id: str):
        """Generate a configuration dictionary for a given thread ID."""
        return {"configurable": {"thread_id": "thread_" + thread_id}}

    def get_overflow_memory_middleware(self, method: str = "trim"):
        """Return the overflow memory handle for a given method."""
        if method == OverflowMemoryMethod.DELETE.value:
            return delete_old_messages
        
        return trim_messages


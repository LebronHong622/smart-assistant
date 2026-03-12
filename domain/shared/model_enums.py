from enum import Enum


class ModelType(Enum):
    """模型类型枚举"""
    CHAT = "chat"  # 基础对话模型
    TOOL_ENABLED_CHAT = "tool_enabled_chat"  # 带工具调用的对话模型
    EMBEDDING = "embedding"  # 嵌入模型


class RoutingStrategy(Enum):
    """路由策略枚举"""
    DEFAULT = "default"  # 默认策略
    PRIORITY = "priority"  # 优先级策略（预留）
    COST_AWARE = "cost_aware"  # 成本感知策略（预留）
    LOAD_BALANCING = "load_balancing"  # 负载均衡策略（预留）
    PERFORMANCE = "performance"  # 性能优先策略（预留）

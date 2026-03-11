from enum import Enum


class OverflowMemoryMethod(Enum):
    TRIM = "trim"
    SUMMARY = "summary"
    DELETE = "delete"


class StorageBackend(str, Enum):
    """存储后端类型"""
    IN_MEMORY = "in_memory"
    REDIS = "redis"


class Framework(str, Enum):
    """AI框架类型"""
    LANGCHAIN = "langchain"
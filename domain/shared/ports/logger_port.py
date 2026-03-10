"""
日志端口 - 定义日志接口
"""

from abc import ABC, abstractmethod
from typing import Any


class LoggerPort(ABC):
    """日志接口"""

    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        """记录信息日志"""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs: Any) -> None:
        """记录错误日志"""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        """记录警告日志"""
        pass

    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        """记录调试日志"""
        pass

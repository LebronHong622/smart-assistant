"""
日志适配器 - 实现日志端口
"""

from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log import app_logger


class LoggerAdapter(LoggerPort):
    """日志适配器实现"""

    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""
        app_logger.info(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """记录错误日志"""
        app_logger.error(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""
        app_logger.warning(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""
        app_logger.debug(message, **kwargs)

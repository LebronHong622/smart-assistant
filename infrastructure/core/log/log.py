"""
日志配置管理类

"""

import sys
import os
from pathlib import Path
from loguru import logger

from config.settings import settings

class LoggerManager:
    """
    提供统一的日志配置和管理功能
    支持文件输出、控制台输出和结构化日志
    单例模式实现
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._log_dir = self._create_log_dir()
            self._setup_logger()
            self._initialized = True

    def _create_log_dir(self) -> str:
        """
        创建日志文件夹

        :return: 日志文件夹路径
        """
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        return str(log_dir)

    def _setup_logger(self):
        """
        配置日志
        """
        # 清除默认的日志处理器
        logger.remove()

        # 配置控制台输出
        logger.add(
            sink=sys.stderr,
            level=settings.app.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )

        # 配置app前缀的日志文件
        logger.add(
            sink=os.path.join(self._log_dir, "app_{time:YYYY-MM-DD}.log"),
            level=settings.app.log_level,
            rotation="00:00",  # 每天轮转
            retention="7 days",  # 7天删除
            compression="zip",  # 压缩旧日志
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8"
        )

        # 配置error前缀的日志文件
        logger.add(
            sink=os.path.join(self._log_dir, "error_{time:YYYY-MM-DD}.log"),
            level="ERROR",
            rotation="00:00",  # 每天轮转
            retention="14 days",  # 14天删除
            compression="zip",  # 压缩旧日志
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8"
        )

        # 配置api前缀的日志文件
        logger.add(
            sink=os.path.join(self._log_dir, "api_{time:YYYY-MM-DD}.log"),
            level="INFO",
            rotation="00:00",  # 每天轮转
            retention="7 days",  # 7天删除
            compression="zip",  # 压缩旧日志
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8"
        )

    def get_logger(self, name: str = None):
        """
        获取配置好的logger实例

        :return: logger实例
        """
        if name:
            return logger.bind(name=name)
        return logger


# 创建单例实例
logger_manager = LoggerManager()
# 导出logger实例供其他模块使用
app_logger = logger_manager.get_logger("app")

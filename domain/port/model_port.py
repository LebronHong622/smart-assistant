"""
模型端口 - 定义模型接口
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelPort(ABC):
    """模型接口"""

    @abstractmethod
    def get_default_model(self) -> Any:
        """获取默认模型"""
        pass

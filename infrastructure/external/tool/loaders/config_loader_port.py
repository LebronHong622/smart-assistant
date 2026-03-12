"""
工具配置加载器接口 - 基础设施层内部接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List


class ToolConfigLoaderPort(ABC):
    """工具配置加载器接口"""

    @abstractmethod
    def load_config(self) -> Dict[str, List[str]]:
        """加载工具配置，路径由具体实现类内部管理"""
        pass
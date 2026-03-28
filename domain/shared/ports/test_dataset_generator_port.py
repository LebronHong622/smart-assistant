"""
测试数据集生成器端口
定义基于RAG文档生成测试数据集的抽象接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class TestDatasetGenerationConfig:
    """测试数据集生成配置"""
    
    def __init__(
        self,
        test_size: int = 10,
        distribution: str = "simple",
        # RAGAS 特定配置
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        # 可选的自定义配置
        extra_config: Optional[Dict[str, Any]] = None
    ):
        self.test_size = test_size
        self.distribution = distribution
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extra_config = extra_config or {}


class ITestDatasetGenerator(ABC):
    """测试数据集生成器端口

    定义从RAG文档生成测试数据集的抽象接口
    具体实现由基础设施层提供（如Ragas适配器）
    """

    @abstractmethod
    def generate_from_documents(
        self,
        documents: List[Any],
        config: TestDatasetGenerationConfig
    ) -> pd.DataFrame:
        """从文档列表生成测试数据集

        Args:
            documents: 文档列表，具体类型由实现决定
            config: 生成配置

        Returns:
            生成的测试数据集DataFrame，标准列包含：
            - question: 问题
            - contexts: 上下文列表
            - ground_truth: 标准答案
            - evolution_type: 问题进化类型（可选）
        """
        pass

    @abstractmethod
    def validate_generated_dataset(
        self,
        df: pd.DataFrame
    ) -> tuple[bool, List[str]]:
        """验证生成的数据集格式是否正确

        Args:
            df: 生成的数据集

        Returns:
            (是否有效, 错误信息列表)
        """
        pass

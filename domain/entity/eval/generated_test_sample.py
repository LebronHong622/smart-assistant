"""
生成的测试样本实体
表示从文档自动生成的单个测试用例和数据集
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class GeneratedTestSample(BaseModel):
    """单个生成的测试样本

    表示从RAG文档自动生成的一个测试问题，包含：
    - question: 生成的问题
    - contexts: 问题使用的上下文文档片段列表
    - ground_truth: 标准答案
    - episode_done: 对话是否结束（可选，用于多轮对话场景）
    """
    question: str
    contexts: List[str]
    ground_truth: str
    episode_done: Optional[bool] = None


class GeneratedTestDataset(BaseModel):
    """生成的测试数据集

    包含多个生成的测试样本，以及可选的元数据
    """
    samples: List[GeneratedTestSample]
    metadata: Dict = Field(default_factory=dict)

    @property
    def count(self) -> int:
        """获取样本数量"""
        return len(self.samples)

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

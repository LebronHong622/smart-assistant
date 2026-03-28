"""
评测结果仓储接口
严格遵循用户要求的接口定义
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entity.eval.eval_result import EvalResult


class IEvalResultRepository(ABC):
    """评测结果仓储接口

    核心规则：
    - 只允许新增/查询，禁止修改/删除
    - 保证评测结果不可篡改，完全可信
    """

    @abstractmethod
    def save_result(self, result: EvalResult) -> EvalResult:
        """保存评测结果（只允许新增，不可修改）"""
        pass

    @abstractmethod
    def save_batch(self, results: List[EvalResult]) -> List[EvalResult]:
        """批量保存评测结果"""
        pass

    @abstractmethod
    def query_by_task(self, task_id: str) -> List[EvalResult]:
        """根据任务ID查询所有结果"""
        pass

    @abstractmethod
    def query_by_version(self, dataset_id: str, dataset_version: str, model_version: str) -> List[EvalResult]:
        """根据数据集版本和模型版本查询结果

        支持全链路追溯：特定数据集版本 + 特定模型版本 的所有评测结果
        """
        pass

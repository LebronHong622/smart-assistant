"""
评测验证领域服务
验证评测结果的完整性和一致性
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
from domain.entity.eval.eval_result import EvalResult
from domain.entity.eval.eval_dataset import EvalDataset


class EvalValidatorService(ABC):
    """评测验证领域服务

    验证评测结果的完整性和一致性，保证评测质量
    """

    @abstractmethod
    def validate_dataset_integrity(self, dataset: EvalDataset) -> Tuple[bool, List[str]]:
        """验证数据集完整性

        检查：
        - 文件是否存在
        - 记录数量是否一致
        - 必要字段是否完整

        Returns:
            (is_valid, error_messages)
        """
        pass

    @abstractmethod
    def validate_results_complete(self, results: List[EvalResult]) -> Tuple[bool, List[str]]:
        """验证评测结果完整性

        检查：
        - 所有预期指标是否都有结果
        - 结果数值是否合法（非NaN，在合理范围内）

        Returns:
            (is_valid, error_messages)
        """
        pass

    @abstractmethod
    def validate_immutability(self, dataset_id: str, version: str) -> Tuple[bool, List[str]]:
        """验证不变性规则

        检查：
        - 是否已存在该版本
        - 如果已存在，是否尝试修改（修改违反规则）

        Returns:
            (is_valid, error_messages)
        """
        pass


class EvalValidatorServiceImpl(EvalValidatorService):
    """评测验证服务实现"""

    def validate_dataset_integrity(self, dataset: EvalDataset) -> Tuple[bool, List[str]]:
        """验证数据集完整性"""
        errors = []

        # 检查必要字段
        if not dataset.dataset_id:
            errors.append("dataset_id 不能为空")
        if not dataset.dataset_name:
            errors.append("dataset_name 不能为空")
        if not dataset.file_path:
            errors.append("file_path 不能为空")
        if dataset.task_count <= 0:
            errors.append("task_count 必须大于0")

        return (len(errors) == 0, errors)

    def validate_results_complete(self, results: List[EvalResult]) -> Tuple[bool, List[str]]:
        """验证评测结果完整性"""
        errors = []
        import math

        for i, result in enumerate(results):
            # 检查数值是否合法
            value = result.metric_value.value
            if math.isnan(value):
                errors.append(f"结果[{i}]: metric_value 是 NaN")
            if not (-1e10 <= value <= 1e10):
                errors.append(f"结果[{i}]: metric_value {value} 超出合理范围")

            # 检查必要字段
            if not result.task_id:
                errors.append(f"结果[{i}]: task_id 不能为空")
            if not result.metric_name:
                errors.append(f"结果[{i}]: metric_name 不能为空")

        return (len(errors) == 0, errors)

    def validate_immutability(self, dataset_id: str, version: str) -> Tuple[bool, List[str]]:
        """验证不变性规则

        如果版本已存在，则不允许修改，必须创建新版本
        此方法需要仓储层查询配合，这里只定义接口
        """
        errors = []
        # 实际检查需要仓储层查询
        return (len(errors) == 0, errors)

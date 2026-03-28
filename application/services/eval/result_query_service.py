"""
评测结果查询应用服务
聚合评测结果查询和统计
"""
from typing import List, Dict, Optional
from domain.entity.eval.eval_result import EvalResult
from domain.repository.eval.i_eval_result_repository import IEvalResultRepository
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class ResultQueryService:
    """评测结果查询应用服务

    提供：
    - 按任务查询
    - 按版本查询
    - 统计聚合
    """

    def __init__(
        self,
        result_repository: IEvalResultRepository,
        logger: Optional[LoggerPort] = None
    ):
        self.result_repository = result_repository
        self.logger = logger or get_app_logger()

    def get_by_task(self, task_id: str) -> List[EvalResult]:
        """按任务ID查询所有结果"""
        return self.result_repository.query_by_task(task_id)

    def get_by_version(
        self,
        dataset_id: str,
        dataset_version: str,
        model_version: str
    ) -> List[EvalResult]:
        """按数据集版本和模型版本查询结果"""
        return self.result_repository.query_by_version(
            dataset_id, dataset_version, model_version
        )

    def aggregate_by_metric(self, results: List[EvalResult]) -> Dict[str, Dict]:
        """按指标聚合结果

        返回每个指标的最新值和统计信息
        """
        aggregated = {}
        for result in results:
            aggregated[result.metric_name] = {
                "value": result.metric_value.value,
                "confidence_lower": result.metric_value.confidence_lower,
                "confidence_upper": result.metric_value.confidence_upper,
                "result_id": result.result_id,
                "task_id": result.task_id,
                "create_time": result.create_time.isoformat() if result.create_time else None,
                "details": result.details
            }
        return aggregated

    def compare_models(
        self,
        dataset_id: str,
        dataset_version: str,
        model_versions: List[str]
    ) -> Dict[str, Dict]:
        """比较多个模型版本在同一数据集上的评测结果

        Returns:
            {
                "metric_name": {
                    "model_v1": value,
                    "model_v2": value
                }
            }
        """
        comparison = {}

        for model_version in model_versions:
            results = self.get_by_version(dataset_id, dataset_version, model_version)
            aggregated = self.aggregate_by_metric(results)

            for metric_name, data in aggregated.items():
                if metric_name not in comparison:
                    comparison[metric_name] = {}
                comparison[metric_name][model_version] = data["value"]

        return comparison

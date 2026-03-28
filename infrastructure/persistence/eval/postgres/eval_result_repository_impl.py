"""
评测结果仓储PostgreSQL实现
强制规则：只允许新增/查询，禁止修改/删除，保证结果可信
"""
from typing import List
from uuid import uuid4
from domain.entity.eval.eval_result import EvalResult
from domain.repository.eval.i_eval_result_repository import IEvalResultRepository
from domain.vo.eval.metric_value import MetricValue
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.persistence.database.postgres_client import PostgreSQLClient
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class EvalResultRepositoryImpl(IEvalResultRepository):
    """评测结果仓储PostgreSQL实现

    核心强制规则：
    - 只允许INSERT，不允许UPDATE/DELETE
    - 保证评测结果不可篡改，完全可信
    - 支持全链路追溯：task_id + dataset_version + model_version
    """

    def __init__(
        self,
        logger: Optional[LoggerPort] = None
    ):
        self.logger = logger or get_app_logger()
        self._client = PostgreSQLClient.get_instance()

    def save_result(self, result: EvalResult) -> EvalResult:
        """保存评测结果（只允许新增，不可修改）

        如果result_id未设置，自动生成UUID
        """
        if not result.result_id:
            result.result_id = str(uuid4())

        self.logger.info(
            f"保存评测结果: result_id={result.result_id}, task_id={result.task_id}, metric={result.metric_name}"
        )

        sql_insert = text("""
            INSERT INTO eval_results (
                result_id, task_id, dataset_id, dataset_version, model_version,
                metric_name, metric_value, confidence_lower, confidence_upper, details
            ) VALUES (
                :result_id, :task_id, :dataset_id, :dataset_version, :model_version,
                :metric_name, :metric_value, :confidence_lower, :confidence_upper, :details
            )
            RETURNING id
        """)

        with self._client.transaction() as conn:
            result_row = conn.execute(sql_insert, {
                "result_id": result.result_id,
                "task_id": result.task_id,
                "dataset_id": result.dataset_id,
                "dataset_version": result.dataset_version,
                "model_version": result.model_version,
                "metric_name": result.metric_name,
                "metric_value": result.metric_value.value,
                "confidence_lower": result.metric_value.confidence_lower,
                "confidence_upper": result.metric_value.confidence_upper,
                "details": result.details
            })

            result.id = result_row.scalar_one()

            self.logger.info(
                f"评测结果保存成功: id={result.id}, result_id={result.result_id}"
            )

            return result

    def save_batch(self, results: List[EvalResult]) -> List[EvalResult]:
        """批量保存评测结果"""
        self.logger.info(f"批量保存评测结果，数量: {len(results)}")

        saved_results = []
        for result in results:
            saved = self.save_result(result)
            saved_results.append(saved)

        return saved_results

    def query_by_task(self, task_id: str) -> List[EvalResult]:
        """根据任务ID查询所有结果"""
        sql = text("""
            SELECT id, result_id, task_id, dataset_id, dataset_version, model_version,
                   metric_name, metric_value, confidence_lower, confidence_upper, details, create_time
            FROM eval_results
            WHERE task_id = :task_id
            ORDER BY metric_name
        """)

        with self._client.connection() as conn:
            result = conn.execute(sql, {"task_id": task_id})
            return self._rows_to_eval_results(result)

    def query_by_version(
        self,
        dataset_id: str,
        dataset_version: str,
        model_version: str
    ) -> List[EvalResult]:
        """根据数据集版本和模型版本查询结果"""
        sql = text("""
            SELECT id, result_id, task_id, dataset_id, dataset_version, model_version,
                   metric_name, metric_value, confidence_lower, confidence_upper, details, create_time
            FROM eval_results
            WHERE dataset_id = :dataset_id
              AND dataset_version = :dataset_version
              AND model_version = :model_version
            ORDER BY metric_name
        """)

        with self._client.connection() as conn:
            result = conn.execute(sql, {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "model_version": model_version
            })
            return self._rows_to_eval_results(result)

    def _rows_to_eval_results(self, rows) -> List[EvalResult]:
        """将查询行转换为EvalResult对象"""
        results = []
        for row in rows:
            metric_value = MetricValue(
                value=row.metric_value,
                confidence_lower=row.confidence_lower,
                confidence_upper=row.confidence_upper
            )

            result = EvalResult(
                id=row.id,
                result_id=row.result_id,
                task_id=row.task_id,
                dataset_id=row.dataset_id,
                dataset_version=row.dataset_version,
                model_version=row.model_version,
                metric_name=row.metric_name,
                metric_value=metric_value,
                details=row.details,
                create_time=row.create_time
            )
            results.append(result)

        return results

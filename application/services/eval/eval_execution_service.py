"""
评测执行应用服务
协调评测任务的执行和结果保存
"""
from typing import List, Dict, Optional
from uuid import uuid4
from domain.entity.eval.eval_task import EvalTask, EvalTaskStatus
from domain.entity.eval.eval_result import EvalResult
from domain.entity.eval.eval_vector import EvalVector
from domain.repository.eval.i_eval_dataset_repository import IEvalDatasetRepository
from domain.repository.eval.i_eval_result_repository import IEvalResultRepository
from domain.repository.eval.i_eval_vector_repository import IEvalVectorRepository
from domain.service.eval.metric_calculate_service import MetricCalculateServiceImpl
from domain.vo.eval.metric_value import MetricValue
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class EvalExecutionService:
    """评测执行应用服务

    协调：
    - 评测任务生命周期管理
    - 指标计算
    - 结果保存
    - 向量保存
    """

    def __init__(
        self,
        dataset_repository: IEvalDatasetRepository,
        result_repository: IEvalResultRepository,
        vector_repository: IEvalVectorRepository,
        metric_service: MetricCalculateServiceImpl,
        logger: Optional[LoggerPort] = None
    ):
        self.dataset_repository = dataset_repository
        self.result_repository = result_repository
        self.vector_repository = vector_repository
        self.metric_service = metric_service
        self.logger = logger or get_app_logger()

    def create_task(
        self,
        task_name: str,
        model_version: str,
        dataset_id: str,
        dataset_version: str,
        creator: str,
        parameters: Optional[Dict] = None
    ) -> EvalTask:
        """创建评测任务"""
        task_id = str(uuid4())

        task = EvalTask(
            task_id=task_id,
            task_name=task_name,
            model_version=model_version,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            creator=creator,
            parameters=parameters or {}
        )

        self.logger.info(
            f"创建评测任务: task_id={task_id}, model_version={model_version}, dataset={dataset_id}@{dataset_version}"
        )

        # TODO: 保存任务到数据库
        # 需要添加EvalTaskRepository，当前版本先简化
        return task

    def save_result(
        self,
        task_id: str,
        dataset_id: str,
        dataset_version: str,
        model_version: str,
        metric_name: str,
        metric_value: float,
        confidence_lower: Optional[float] = None,
        confidence_upper: Optional[float] = None,
        details: Optional[Dict] = None
    ) -> EvalResult:
        """保存单个评测结果"""
        result_id = str(uuid4())

        metric = MetricValue(
            value=metric_value,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper
        )

        result = EvalResult(
            result_id=result_id,
            task_id=task_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_version=model_version,
            metric_name=metric_name,
            metric_value=metric,
            details=details
        )

        saved = self.result_repository.save_result(result)
        self.logger.info(
            f"评测结果保存成功: task_id={task_id}, metric={metric_name}, value={metric_value:.4f}"
        )

        return saved

    def save_batch_results(self, results: List[EvalResult]) -> List[EvalResult]:
        """批量保存评测结果"""
        saved = self.result_repository.save_batch(results)
        self.logger.info(f"批量保存评测结果完成，数量: {len(saved)}")
        return saved

    def insert_vector(
        self,
        task_id: str,
        dataset_id: str,
        dataset_version: str,
        record_id: str,
        embedding: List[float],
        content: Optional[str] = None,
        meta_json: Optional[Dict] = None
    ) -> EvalVector:
        """插入向量"""
        vector_id = str(uuid4())

        vector = EvalVector(
            vector_id=vector_id,
            task_id=task_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            record_id=record_id,
            content=content,
            meta_json=meta_json,
            embedding=embedding
        )

        saved = self.vector_repository.insert_vector(vector)
        return saved

    def insert_batch_vectors(self, vectors: List[EvalVector]) -> List[EvalVector]:
        """批量插入向量"""
        return self.vector_repository.insert_batch(vectors)

    def search_similar_vectors(
        self,
        query_embedding: List[float],
        dataset_id: str,
        dataset_version: str,
        limit: int = 10
    ) -> List[EvalVector]:
        """搜索相似向量"""
        return self.vector_repository.search_vector(
            query_embedding, dataset_id, dataset_version, limit
        )

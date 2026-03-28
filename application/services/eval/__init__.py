"""
评测领域应用服务
"""
from application.services.eval.dataset_management_service import DatasetManagementService
from application.services.eval.eval_execution_service import EvalExecutionService
from application.services.eval.result_query_service import ResultQueryService

__all__ = ["DatasetManagementService", "EvalExecutionService", "ResultQueryService"]

"""
评测领域仓储接口
"""
from domain.repository.eval.i_eval_dataset_repository import IEvalDatasetRepository
from domain.repository.eval.i_eval_result_repository import IEvalResultRepository
from domain.repository.eval.i_eval_vector_repository import IEvalVectorRepository

__all__ = ["IEvalDatasetRepository", "IEvalResultRepository", "IEvalVectorRepository"]

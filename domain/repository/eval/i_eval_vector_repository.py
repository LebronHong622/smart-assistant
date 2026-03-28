"""
评测向量仓储接口
严格遵循用户要求的接口定义
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entity.eval.eval_vector import EvalVector


class IEvalVectorRepository(ABC):
    """评测向量仓储接口

    分层存储：
    - Milvus存储实际向量数据
    - PostgreSQL存储元数据
    """

    @abstractmethod
    def insert_vector(self, vector: EvalVector) -> EvalVector:
        """插入向量元数据，并将向量存入Milvus"""
        pass

    @abstractmethod
    def insert_batch(self, vectors: List[EvalVector]) -> List[EvalVector]:
        """批量插入向量元数据"""
        pass

    @abstractmethod
    def search_vector(self, query_embedding: List[float],
                     dataset_id: str, dataset_version: str, limit: int = 10) -> List[EvalVector]:
        """搜索相似向量

        Args:
            query_embedding: 查询向量
            dataset_id: 按数据集ID过滤
            dataset_version: 按数据集版本过滤
            limit: 返回结果数量限制
        """
        pass

    @abstractmethod
    def get_vector_meta(self, vector_id: str) -> Optional[EvalVector]:
        """获取向量元数据"""
        pass

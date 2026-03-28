"""
向量元数据仓储PostgreSQL实现
存储向量元数据到PostgreSQL，实际向量存储在Milvus
"""
from typing import List, Optional
from uuid import uuid4
from domain.entity.eval.eval_vector import EvalVector
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.persistence.database.postgres_client import PostgreSQLClient
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class EvalVectorPostgresRepositoryImpl:
    """向量元数据PostgreSQL仓储实现

    职责：
    - 存储向量元数据
    - 关联到任务、数据集版本
    - 提供过滤查询能力
    """

    def __init__(
        self,
        logger: Optional[LoggerPort] = None
    ):
        self.logger = logger or get_app_logger()
        self._client = PostgreSQLClient.get_instance()

    def insert_vector(self, vector: EvalVector) -> EvalVector:
        """插入向量元数据"""
        if not vector.vector_id:
            vector.vector_id = str(uuid4())

        sql_insert = text("""
            INSERT INTO eval_vectors (
                vector_id, milvus_id, task_id, dataset_id, dataset_version,
                record_id, content, meta_json
            ) VALUES (
                :vector_id, :milvus_id, :task_id, :dataset_id, :dataset_version,
                :record_id, :content, :meta_json
            )
            RETURNING id
        """)

        with self._client.transaction() as conn:
            result = conn.execute(sql_insert, {
                "vector_id": vector.vector_id,
                "milvus_id": vector.milvus_id,
                "task_id": vector.task_id,
                "dataset_id": vector.dataset_id,
                "dataset_version": vector.dataset_version,
                "record_id": vector.record_id,
                "content": vector.content,
                "meta_json": vector.meta_json
            })

            vector.id = result.scalar_one()
            self.logger.debug(f"向量元数据插入成功: id={vector.id}, vector_id={vector.vector_id}")

            return vector

    def insert_batch(self, vectors: List[EvalVector]) -> List[EvalVector]:
        """批量插入向量元数据"""
        saved_vectors = []
        for vector in vectors:
            saved = self.insert_vector(vector)
            saved_vectors.append(saved)
        return saved_vectors

    def get_by_vector_id(self, vector_id: str) -> Optional[EvalVector]:
        """根据vector_id查询元数据"""
        sql = text("""
            SELECT id, vector_id, milvus_id, task_id, dataset_id, dataset_version,
                   record_id, content, meta_json, create_time
            FROM eval_vectors
            WHERE vector_id = :vector_id
        """)

        with self._client.connection() as conn:
            result = conn.execute(sql, {"vector_id": vector_id})
            row = result.first()

            if row is None:
                return None

            return EvalVector(
                id=row.id,
                vector_id=row.vector_id,
                milvus_id=row.milvus_id,
                task_id=row.task_id,
                dataset_id=row.dataset_id,
                dataset_version=row.dataset_version,
                record_id=row.record_id,
                content=row.content,
                meta_json=row.meta_json,
                create_time=row.create_time
            )

    def list_by_dataset(self, dataset_id: str, dataset_version: str) -> List[EvalVector]:
        """按数据集和版本查询所有向量元数据"""
        sql = text("""
            SELECT id, vector_id, milvus_id, task_id, dataset_id, dataset_version,
                   record_id, content, meta_json, create_time
            FROM eval_vectors
            WHERE dataset_id = :dataset_id AND dataset_version = :dataset_version
            ORDER BY create_time
        """)

        with self._client.connection() as conn:
            result = conn.execute(sql, {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version
            })
            return self._rows_to_eval_vectors(result)

    def _rows_to_eval_vectors(self, rows) -> List[EvalVector]:
        """转换行到EvalVector列表"""
        vectors = []
        for row in rows:
            vector = EvalVector(
                id=row.id,
                vector_id=row.vector_id,
                milvus_id=row.milvus_id,
                task_id=row.task_id,
                dataset_id=row.dataset_id,
                dataset_version=row.dataset_version,
                record_id=row.record_id,
                content=row.content,
                meta_json=row.meta_json,
                create_time=row.create_time
            )
            vectors.append(vector)
        return vectors

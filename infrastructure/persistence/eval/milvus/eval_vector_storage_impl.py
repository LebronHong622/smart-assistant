"""
评测向量存储Milvus实现
Milvus存储实际向量数据，MySQL存储元数据
"""
from typing import List, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Index
from config.settings import settings
from domain.entity.eval.eval_vector import EvalVector
from domain.repository.eval.i_eval_vector_repository import IEvalVectorRepository
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.persistence.eval.mysql.eval_vector_repository_impl import EvalVectorMySQLRepositoryImpl
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class EvalVectorStorageImpl(IEvalVectorRepository):
    """评测向量存储Milvus实现

    分层存储：
    - Milvus: 存储实际向量embedding
    - MySQL: 存储元数据（关联task_id, dataset_id, version等）
    """

    COLLECTION_NAME = "eval_embeddings"

    def __init__(
        self,
        meta_repository: EvalVectorMySQLRepositoryImpl,
        logger: Optional[LoggerPort] = None
    ):
        self.meta_repository = meta_repository
        self.logger = logger or get_app_logger()
        self._collection = self._init_collection()

    def _init_collection(self) -> Collection:
        """初始化Milvus集合，如果不存在则创建"""
        # 获取连接
        from pymilvus import connections
        if connections.has_connection():
            connections.connect(
                uri=settings.milvus.milvus_uri,
                token=settings.milvus.milvus_token if hasattr(settings.milvus, 'milvus_token') else None
            )
        else:
            connections.connect(
                uri=settings.milvus.milvus_uri,
                token=settings.milvus.milvus_token if hasattr(settings.milvus, 'milvus_token') else None
            )

        # 检查集合是否存在
        from pymilvus import utility
        if utility.has_collection(self.COLLECTION_NAME):
            self.logger.info(f"Milvus集合已存在: {self.COLLECTION_NAME}")
            collection = Collection(self.COLLECTION_NAME)
        else:
            self.logger.info(f"创建Milvus集合: {self.COLLECTION_NAME}")
            collection = self._create_collection()

        # 加载集合到内存
        collection.load()
        return collection

    def _create_collection(self) -> Collection:
        """创建集合schema"""
        # 字段定义
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="dataset_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="dataset_version", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus.milvus_dimension)
        ]

        schema = CollectionSchema(fields, description="评测向量集合")
        collection = Collection(self.COLLECTION_NAME, schema)

        # 创建索引
        index_params = {
            "index_type": settings.milvus.milvus_index_type,
            "metric_type": settings.milvus.milvus_metric_type,
            "params": {"nlist": settings.milvus.milvus_n_list}
        }
        index = Index(collection, "embedding", index_params)
        self.logger.info(f"Milvus集合创建完成，索引类型: {settings.milvus.milvus_index_type}")

        return collection

    def insert_vector(self, vector: EvalVector) -> EvalVector:
        """插入向量

        - 向量插入Milvus
        - 元数据插入PostgreSQL
        """
        if not vector.has_embedding:
            raise ValueError("向量数据不能为空")

        # 插入到Milvus
        milvus_id = vector.vector_id
        insert_data = [{
            "id": vector.vector_id,
            "vector_id": vector.vector_id,
            "dataset_id": vector.dataset_id,
            "dataset_version": vector.dataset_version,
            "embedding": vector.embedding
        }]

        mr = self._collection.insert(insert_data)
        self.logger.debug(f"向量插入Milvus成功: vector_id={vector.vector_id}")

        # 更新milvus_id
        vector.milvus_id = str(mr.primary_keys[0])

        # 插入元数据到PostgreSQL
        vector = self.meta_repository.insert_vector(vector)

        # 刷新索引
        self._collection.flush()

        return vector

    def insert_batch(self, vectors: List[EvalVector]) -> List[EvalVector]:
        """批量插入向量"""
        self.logger.info(f"批量插入向量，数量: {len(vectors)}")

        # 分批插入Milvus
        milvus_data = []
        for vector in vectors:
            if vector.has_embedding:
                milvus_data.append({
                    "id": vector.vector_id,
                    "vector_id": vector.vector_id,
                    "dataset_id": vector.dataset_id,
                    "dataset_version": vector.dataset_version,
                    "embedding": vector.embedding
                })

        if milvus_data:
            mr = self._collection.insert(milvus_data)
            self._collection.flush()
            self.logger.info(f"批量插入Milvus完成，插入数量: {len(milvus_data)}")

            # 更新milvus_id
            for i, pk in enumerate(mr.primary_keys):
                vectors[i].milvus_id = str(pk)

        # 插入元数据到PostgreSQL
        saved_vectors = self.meta_repository.insert_batch(vectors)
        return saved_vectors

    def search_vector(
        self,
        query_embedding: List[float],
        dataset_id: str,
        dataset_version: str,
        limit: int = 10
    ) -> List[EvalVector]:
        """搜索相似向量

        使用dataset_id和dataset_version过滤，只在特定版本数据中搜索
        """
        # 构造过滤表达式
        expr = f'dataset_id == "{dataset_id}" && dataset_version == "{dataset_version}"'

        # 搜索
        search_params = {
            "metric_type": settings.milvus.milvus_metric_type,
            "params": {"nprobe": 10}
        }

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["vector_id"]
        )

        # 获取结果对应的元数据
        result_vectors = []
        for hit in results[0]:
            vector_id = hit.entity.get("vector_id")
            meta = self.meta_repository.get_by_vector_id(vector_id)
            if meta:
                meta.distance = hit.distance
                result_vectors.append(meta)

        self.logger.debug(
            f"向量搜索完成: dataset_id={dataset_id}, version={dataset_version}, "
            f"返回结果数量: {len(result_vectors)}"
        )

        return result_vectors

    def get_vector_meta(self, vector_id: str) -> Optional[EvalVector]:
        """获取向量元数据"""
        return self.meta_repository.get_by_vector_id(vector_id)

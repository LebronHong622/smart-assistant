"""
测试数据集仓储MySQL实现
实现不可修改规则：总是创建新版本，旧版本标记为deprecated
"""
from typing import List, Optional
from sqlalchemy import text
from domain.entity.eval.eval_dataset import EvalDataset
from domain.repository.eval.i_eval_dataset_repository import IEvalDatasetRepository
from domain.vo.eval.version import Version
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.persistence.database.mysql_client import MySQLClient
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class EvalDatasetRepositoryImpl(IEvalDatasetRepository):
    """测试数据集仓储MySQL实现

    核心规则强制：
    1. create_dataset 从不修改已有版本，只插入新版本
    2. 创建新版本时自动将同一dataset_id的其他活跃版本标记为deprecated
    3. 禁止直接修改已有版本
    """

    def __init__(self, logger: Optional[LoggerPort] = None):
        self.logger = logger or get_app_logger()
        self._client = MySQLClient.get_instance()

    def create_dataset(self, dataset: EvalDataset) -> EvalDataset:
        """创建新数据集版本

        - 插入新记录到数据库
        - 将同一dataset_id的其他活跃版本标记为deprecated
        """
        version_str = dataset.version.to_string()
        self.logger.info(
            f"创建新数据集版本: dataset_id={dataset.dataset_id}, version={version_str}"
        )

        # 检查版本是否已存在
        existing = self.get_by_version(dataset.dataset_id, dataset.version)
        if existing is not None:
            raise ValueError(
                f"版本已存在: {dataset.dataset_id}/{version_str}，"
                "不能修改已发布版本，请创建新版本"
            )

        # 开始事务
        with self._client.get_session() as session:
            # 将所有旧的活跃版本标记为deprecated
            if dataset.id is None:
                sql_deprecate = text("""
                    UPDATE eval_datasets
                    SET status = 'deprecated'
                    WHERE dataset_id = :dataset_id AND status = 'active'
                """)
                result = session.execute(sql_deprecate, {
                    "dataset_id": dataset.dataset_id
                })
                if result.rowcount > 0:
                    self.logger.info(
                        f"已将 {result.rowcount} 个旧版本标记为deprecated: {dataset.dataset_id}"
                    )

            # 插入新版本
            sql_insert = text("""
                INSERT INTO eval_datasets (
                    dataset_id, dataset_name, version, file_path,
                    create_time, update_time, creator, updater, status, metadata, task_count
                ) VALUES (
                    :dataset_id, :dataset_name, :version, :file_path,
                    :create_time, :update_time, :creator, :updater, :status, :metadata, :task_count
                )
            """)

            result = session.execute(sql_insert, {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.dataset_name,
                "version": version_str,
                "file_path": dataset.file_path,
                "create_time": dataset.create_time,
                "update_time": dataset.update_time,
                "creator": dataset.creator,
                "updater": dataset.updater,
                "status": dataset.status.value,
                "metadata": dataset.metadata,
                "task_count": dataset.task_count
            })

            dataset_id_db = result.lastrowid
            dataset.id = dataset_id_db

            self.logger.info(
                f"新数据集版本创建成功: id={dataset.id}, dataset_id={dataset.dataset_id}, version={version_str}"
            )

            return dataset

    def get_by_version(self, dataset_id: str, version: Version) -> Optional[EvalDataset]:
        """根据数据集ID和版本查询"""
        version_str = version.to_string()

        sql = text("""
            SELECT id, dataset_id, dataset_name, version, file_path,
                   create_time, update_time, creator, updater, status, metadata, task_count
            FROM eval_datasets
            WHERE dataset_id = :dataset_id AND version = :version
        """)

        with self._client.get_session() as session:
            result = session.execute(sql, {
                "dataset_id": dataset_id,
                "version": version_str
            })
            row = result.first()

            if row is None:
                return None

            return EvalDataset(
                id=row.id,
                dataset_id=row.dataset_id,
                dataset_name=row.dataset_name,
                version=Version.parse(row.version),
                file_path=row.file_path,
                create_time=row.create_time,
                update_time=row.update_time,
                creator=row.creator,
                updater=row.updater,
                status=row.status,
                metadata=row.metadata,
                task_count=row.task_count
            )

    def get_latest(self, dataset_id: str) -> Optional[EvalDataset]:
        """获取最新活跃版本

        按版本号排序，返回最大的活跃版本
        """
        sql = text("""
            SELECT id, dataset_id, dataset_name, version, file_path,
                   create_time, update_time, creator, updater, status, metadata, task_count
            FROM eval_datasets
            WHERE dataset_id = :dataset_id AND status = 'active'
            ORDER BY version DESC
            LIMIT 1
        """)

        with self._client.get_session() as session:
            result = session.execute(sql, {"dataset_id": dataset_id})
            row = result.first()

            if row is None:
                return None

            return EvalDataset(
                id=row.id,
                dataset_id=row.dataset_id,
                dataset_name=row.dataset_name,
                version=Version.parse(row.version),
                file_path=row.file_path,
                create_time=row.create_time,
                update_time=row.update_time,
                creator=row.creator,
                updater=row.updater,
                status=row.status,
                metadata=row.metadata,
                task_count=row.task_count
            )

    def list_versions(self, dataset_id: str) -> List[EvalDataset]:
        """列出所有版本"""
        sql = text("""
            SELECT id, dataset_id, dataset_name, version, file_path,
                   create_time, update_time, creator, updater, status, metadata, task_count
            FROM eval_datasets
            WHERE dataset_id = :dataset_id
            ORDER BY version DESC
        """)

        with self._client.get_session() as session:
            result = session.execute(sql, {"dataset_id": dataset_id})
            datasets = []

            for row in result:
                dataset = EvalDataset(
                    id=row.id,
                    dataset_id=row.dataset_id,
                    dataset_name=row.dataset_name,
                    version=Version.parse(row.version),
                    file_path=row.file_path,
                    create_time=row.create_time,
                    update_time=row.update_time,
                    creator=row.creator,
                    updater=row.updater,
                    status=row.status,
                    metadata=row.metadata,
                    task_count=row.task_count
                )
                datasets.append(dataset)

            return datasets

"""
数据集管理应用服务
协调数据集的创建、版本管理和查询
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from domain.entity.eval.eval_dataset import EvalDataset
from domain.repository.eval.i_eval_dataset_repository import IEvalDatasetRepository
from domain.repository.eval.i_dataset_file_storage import IDatasetFileStorage
from domain.service.eval.dataset_version_service import DatasetVersionService
from domain.vo.eval.version import Version
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class DatasetManagementService:
    """数据集管理应用服务

    协调：
    - 新版本创建
    - 文件存储
    - 数据库版本管理
    - 查询
    """

    def __init__(
        self,
        dataset_repository: IEvalDatasetRepository,
        version_service: DatasetVersionService,
        file_storage: IDatasetFileStorage,
        logger: Optional[LoggerPort] = None
    ):
        self.dataset_repository = dataset_repository
        self.version_service = version_service
        self.file_storage = file_storage
        self.logger = logger or get_app_logger()

    def create_from_dataframe(
        self,
        dataset_id: str,
        dataset_name: str,
        creator: str,
        df: pd.DataFrame,
        is_major_change: bool = False,
        format: str = "parquet",
        metadata: Optional[Dict] = None
    ) -> Tuple[EvalDataset, str]:
        """从DataFrame创建新版本数据集

        Args:
            dataset_id: 数据集业务ID
            dataset_name: 数据集名称
            creator: 创建者
            df: 数据
            is_major_change: 是否为重大变更，影响版本升级
            format: 文件格式

        Returns:
            (创建好的数据集对象, 版本字符串)
        """
        # 获取最新版本，生成下一个版本
        latest = self.dataset_repository.get_latest(dataset_id)
        if latest:
            self.logger.info(f"发现现有数据集，最新版本: {latest.version.to_string()}")
            next_version = self.version_service.generate_next_version(
                latest.version, is_major_change
            )
        else:
            self.logger.info(f"创建全新数据集: {dataset_id}")
            next_version = Version(major=1, minor=0)

        version_str = next_version.to_string()

        # 保存文件
        filename = dataset_id
        file_path = self.file_storage.save_dataframe(
            dataset_id, version_str, df, filename, format
        )

        # 创建数据集实体
        dataset = EvalDataset(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            version=next_version,
            file_path=file_path,
            creator=creator,
            task_count=len(df),
            metadata=metadata or {}
        )

        # 保存到数据库
        dataset = self.dataset_repository.create_dataset(dataset)

        self.logger.info(
            f"新版本数据集创建完成: dataset_id={dataset_id}, version={version_str}, "
            f"task_count={len(df)}, file={file_path}"
        )

        return dataset, version_str

    def get_by_version(self, dataset_id: str, version_str: str) -> Optional[EvalDataset]:
        """获取特定版本数据集"""
        version = Version.parse(version_str)
        return self.dataset_repository.get_by_version(dataset_id, version)

    def get_latest(self, dataset_id: str) -> Optional[EvalDataset]:
        """获取最新版本"""
        return self.dataset_repository.get_latest(dataset_id)

    def list_versions(self, dataset_id: str) -> List[EvalDataset]:
        """列出所有版本"""
        return self.dataset_repository.list_versions(dataset_id)

    def load_dataframe(self, dataset: EvalDataset) -> pd.DataFrame:
        """加载数据集DataFrame

        优先从本地文件读取
        """
        return self.file_storage.load_dataframe(dataset.file_path)

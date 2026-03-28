"""
数据集文件存储实现
存储测试数据集到本地文件，支持parquet和csv格式
"""
import os
import pandas as pd
from typing import Optional
from domain.repository.eval.i_dataset_file_storage import IDatasetFileStorage
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class DatasetFileStorageImpl(IDatasetFileStorage):
    """数据集文件存储实现

    存储路径规则：
        data/eval/datasets/{dataset_id}/{version}/{filename}.parquet
    实现domain层定义的IDatasetFileStorage接口
    """

    def __init__(
        self,
        base_dir: str = "data/eval/datasets",
        logger: Optional[LoggerPort] = None
    ):
        self.base_dir = base_dir
        self.logger = logger or get_app_logger()

    def _get_file_path(self, dataset_id: str, version_str: str, filename: str) -> str:
        """获取完整文件路径"""
        return os.path.join(self.base_dir, dataset_id, version_str, f"{filename}.parquet")

    def _ensure_dir_exists(self, dataset_id: str, version_str: str) -> str:
        """确保目录存在，返回目录路径"""
        dir_path = os.path.join(self.base_dir, dataset_id, version_str)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def save_dataframe(
        self,
        dataset_id: str,
        version_str: str,
        df: pd.DataFrame,
        filename: str = "data",
        format: str = "parquet"
    ) -> str:
        """保存DataFrame到文件

        Args:
            dataset_id: 数据集ID
            version_str: 版本字符串（vX.Y）
            df: 要保存的DataFrame
            filename: 文件名（不含扩展名）
            format: 文件格式，parquet 或 csv

        Returns:
            保存后的文件完整路径
        """
        dir_path = self._ensure_dir_exists(dataset_id, version_str)
        file_path = os.path.join(dir_path, f"{filename}.{format}")

        self.logger.info(f"正在保存数据集文件: {file_path}, 行数: {len(df)}")

        if format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            df.to_csv(file_path, index=False, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {format}, 支持 parquet 或 csv")

        self.logger.info(f"数据集文件保存成功: {file_path}")
        return file_path

    def load_dataframe(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """从文件加载DataFrame

        Args:
            file_path: 文件完整路径
            format: 文件格式，如果为None则从扩展名推断

        Returns:
            加载的DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")

        if format is None:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            format = ext

        self.logger.info(f"正在加载数据集文件: {file_path}")

        if format == "parquet":
            df = pd.read_parquet(file_path)
        elif format == "csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式: {format}")

        self.logger.info(f"数据集文件加载成功: {file_path}, 行数: {len(df)}")
        return df

    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        return os.path.exists(file_path)

    def delete(self, file_path: str) -> None:
        """删除文件（标记废弃后可选删除）"""
        if os.path.exists(file_path):
            os.remove(file_path)
            self.logger.info(f"已删除数据集文件: {file_path}")

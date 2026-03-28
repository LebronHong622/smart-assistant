"""
数据集文件存储仓储接口
定义文件存储的抽象接口，位于domain层
"""
from abc import ABC, abstractmethod
import pandas as pd


class IDatasetFileStorage(ABC):
    """数据集文件存储接口

    定义保存和加载DataFrame到文件的抽象接口
    具体实现由基础设施层提供
    """

    @abstractmethod
    def save_dataframe(
        self,
        dataset_id: str,
        version_str: str,
        df: pd.DataFrame,
        filename: str = "data",
        format: str = "parquet"
    ) -> str:
        """保存DataFrame到文件

        Returns:
            保存后的完整文件路径
        """
        pass

    @abstractmethod
    def load_dataframe(
        self,
        file_path: str,
        format: str = None
    ) -> pd.DataFrame:
        """从文件加载DataFrame"""
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> None:
        """删除文件"""
        pass

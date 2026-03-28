"""
测试数据集仓储接口
严格遵循用户要求的接口定义
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entity.eval.eval_dataset import EvalDataset
from domain.vo.eval.version import Version


class IEvalDatasetRepository(ABC):
    """测试数据集仓储接口

    核心规则：
    - 从不修改已有版本，只能创建新版本
    - 修改必须生成新版本，旧版本标记为deprecated
    """

    @abstractmethod
    def create_dataset(self, dataset: EvalDataset) -> EvalDataset:
        """创建新数据集版本

        注意：此方法从不修改已有版本，只插入新版本
        如果是同一dataset_id的新版本，会自动将旧版本标记为deprecated
        """
        pass

    @abstractmethod
    def get_by_version(self, dataset_id: str, version: Version) -> Optional[EvalDataset]:
        """根据数据集ID和版本查询特定版本"""
        pass

    @abstractmethod
    def get_latest(self, dataset_id: str) -> Optional[EvalDataset]:
        """获取指定数据集ID的最新活跃版本"""
        pass

    @abstractmethod
    def list_versions(self, dataset_id: str) -> List[EvalDataset]:
        """列出指定数据集的所有版本（包括active和deprecated）"""
        pass

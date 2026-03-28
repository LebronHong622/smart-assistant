"""
数据集版本领域服务
处理版本生成和弃用逻辑
"""
from abc import ABC, abstractmethod
from typing import Optional
from domain.entity.eval.eval_dataset import EvalDataset
from domain.vo.eval.version import Version


class DatasetVersionService(ABC):
    """数据集版本领域服务

    负责版本生成和版本状态管理，保证版本化规则被正确遵守
    """

    @abstractmethod
    def generate_next_version(self, latest_version: Optional[Version], is_major_change: bool = False) -> Version:
        """生成下一个版本

        Args:
            latest_version: 当前最新版本，如果是全新数据集则为None
            is_major_change: 是否为重大变更，如果是则升级主版本，否则升级次版本

        Returns:
            生成的新版本
        """
        pass

    @abstractmethod
    def deprecate_old_versions(self, dataset: EvalDataset) -> None:
        """弃用旧版本

        将同一dataset_id的所有其他活跃版本标记为deprecated
        新版本保持active
        """
        pass

    @abstractmethod
    def validate_version_change(self, existing_dataset: EvalDataset, new_version: Version) -> None:
        """验证版本变更是否合法

        检查是否违反了"不可修改已有版本"的规则

        Raises:
            ValueError: 如果变更不合法
        """
        pass


class DatasetVersionServiceImpl(DatasetVersionService):
    """数据集版本领域服务实现"""

    def generate_next_version(self, latest_version: Optional[Version], is_major_change: bool = False) -> Version:
        """生成下一个版本"""
        if latest_version is None:
            # 全新数据集，从v1.0开始
            return Version(major=1, minor=0)

        if is_major_change:
            return latest_version.next_major()
        else:
            return latest_version.next_minor()

    def deprecate_old_versions(self, dataset: EvalDataset) -> None:
        """弃用旧版本 - 此方法只是业务逻辑，实际数据库更新由仓储完成"""
        # 领域层面标记所有旧版本需要弃用
        # 具体的数据库更新在仓储层实现
        pass

    def validate_version_change(self, existing_dataset: EvalDataset, new_version: Version) -> None:
        """验证版本变更

        如果尝试修改已有版本的内容，抛出异常，必须生成新版本
        """
        if existing_dataset.id is not None:
            # 已有版本不允许修改，必须生成新版本
            raise ValueError(
                f"无法修改已发布版本 {existing_dataset.dataset_id}/{existing_dataset.version}，"
                "请基于此版本创建新版本"
            )

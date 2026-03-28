"""
测试数据集生成应用服务
协调从RAG文档生成测试数据集并保存的完整用例
"""
from typing import List, Any, Optional, Tuple
import pandas as pd

from domain.entity.eval.eval_dataset import EvalDataset
from application.services.eval.dataset_management_service import DatasetManagementService
from domain.shared.ports.test_dataset_generator_port import (
    ITestDatasetGenerator,
    TestDatasetGenerationConfig
)
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class TestDatasetGenerationService:
    """测试数据集生成应用服务

    协调：
    - 调用Ragas生成测试数据集
    - 验证生成结果
    - 保存为新版本数据集（复用DatasetManagementService）
    """

    def __init__(
        self,
        dataset_management_service: DatasetManagementService,
        test_generator: ITestDatasetGenerator,
        logger: Optional[LoggerPort] = None
    ):
        self.dataset_service = dataset_management_service
        self.test_generator = test_generator
        self.logger = logger or get_app_logger()

    def generate_and_save(
        self,
        dataset_id: str,
        dataset_name: str,
        creator: str,
        documents: List[Any],
        test_size: int = 10,
        distribution: str = "simple",
        is_major_change: bool = False,
        format: str = "parquet"
    ) -> Tuple[EvalDataset, str]:
        """生成测试数据集并保存为新版本

        Args:
            dataset_id: 数据集业务ID
            dataset_name: 数据集名称
            creator: 创建者
            documents: RAG文档列表
            test_size: 生成测试用例数量
            distribution: 问题复杂度分布 (simple/complex)
            is_major_change: 是否为重大变更
            format: 文件格式

        Returns:
            (创建的数据集实体, 版本字符串)
        """
        self.logger.info(
            f"开始生成测试数据集: dataset_id={dataset_id}, "
            f"test_size={test_size}, distribution={distribution}"
        )

        # 1. 创建生成配置
        config = TestDatasetGenerationConfig(
            test_size=test_size,
            distribution=distribution
        )

        # 2. 生成测试数据集
        df = self.test_generator.generate_from_documents(documents, config)

        # 3. 验证生成结果
        is_valid, errors = self.test_generator.validate_generated_dataset(df)
        if not is_valid:
            error_msg = f"生成数据集验证失败: {', '.join(errors)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 4. 构建元数据
        metadata = {
            "generation_source": "ragas",
            "test_size": test_size,
            "distribution": distribution,
            "source_documents_count": len(documents)
        }

        # 5. 复用数据集管理服务保存
        dataset, version_str = self.dataset_service.create_from_dataframe(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            creator=creator,
            df=df,
            is_major_change=is_major_change,
            format=format,
            metadata=metadata
        )

        self.logger.info(
            f"测试数据集生成并保存完成: dataset_id={dataset_id}, "
            f"version={version_str}, task_count={len(df)}"
        )

        return dataset, version_str

    def generate_only(
        self,
        documents: List[Any],
        test_size: int = 10,
        distribution: str = "simple"
    ) -> pd.DataFrame:
        """仅生成测试数据集，不保存

        用于预览或调试场景
        """
        config = TestDatasetGenerationConfig(
            test_size=test_size,
            distribution=distribution
        )

        df = self.test_generator.generate_from_documents(documents, config)

        # 验证但不抛出异常
        is_valid, errors = self.test_generator.validate_generated_dataset(df)
        if not is_valid:
            self.logger.warning(f"生成数据集存在警告: {errors}")

        return df

    def preview_generation(
        self,
        documents: List[Any],
        test_size: int = 5
    ) -> dict:
        """预览生成结果（生成少量样本）"""
        df = self.generate_only(documents, test_size=test_size)

        return {
            "total_samples": len(df),
            "columns": list(df.columns),
            "sample_questions": df['question'].head(3).tolist() if 'question' in df.columns else [],
            "preview_data": df.head(3).to_dict('records')
        }

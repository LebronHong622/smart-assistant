"""
测试生成器工厂
从YAML配置创建ITestDatasetGenerator实例
"""
from typing import Optional
from domain.shared.ports.test_dataset_generator_port import ITestDatasetGenerator
from infrastructure.external.eval.adapters.ragas_single_hop_adapter import RagasSingleHopAdapter


class TestGeneratorFactory:
    """测试生成器工厂"""

    _instance: Optional[ITestDatasetGenerator] = None
    _config_path: Optional[str] = None

    @classmethod
    def get_generator(
        cls,
        config_path: str = "config/eval/test_dataset_config.yaml",
    ) -> ITestDatasetGenerator:
        """获取测试生成器实例（单例）

        Args:
            config_path: 配置文件路径

        Returns:
            ITestDatasetGenerator实例
        """
        if cls._instance is None or cls._config_path != config_path:
            cls._instance = RagasSingleHopAdapter(config_path)
            cls._config_path = config_path
        return cls._instance

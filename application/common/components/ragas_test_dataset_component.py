"""
Ragas测试数据集生成组件
注册到应用组件注册表
"""
from application.common.component import Component, ComponentStatus
from domain.shared.ports.test_dataset_generator_port import ITestDatasetGenerator
from infrastructure.external.eval.factories.test_generator_factory import TestGeneratorFactory


class RagasTestDatasetComponent(Component):
    """Ragas测试数据集生成组件"""

    def __init__(
        self,
        config_path: str = "config/eval/test_dataset_config.yaml",
    ):
        super().__init__("ragas_test_dataset_generator")
        self.config_path = config_path
        self._generator: ITestDatasetGenerator = None

    def initialize(self) -> None:
        """初始化组件"""
        self._set_status(ComponentStatus.INITIALIZING)
        try:
            self._generator = TestGeneratorFactory.get_generator(self.config_path)
            self._set_status(ComponentStatus.RUNNING)
        except Exception as e:
            self._set_status(ComponentStatus.FAILED)
            raise

    def shutdown(self) -> None:
        """关闭组件"""
        self._generator = None
        self._set_status(ComponentStatus.STOPPED)

    def health_check(self) -> bool:
        """健康检查"""
        return self._status == ComponentStatus.RUNNING and self._generator is not None

    def get_instance(self) -> ITestDatasetGenerator:
        """获取生成器实例"""
        if self._status != ComponentStatus.RUNNING:
            raise RuntimeError(f"组件状态不正确: {self._status}")
        return self._generator

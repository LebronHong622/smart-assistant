"""
模型适配器 - 实现模型端口
"""

from domain.port.model_port import ModelPort
from infrastructure.model.model_manager import model_manager


class ModelAdapter(ModelPort):
    """模型适配器实现"""

    def __init__(self):
        self._model_manager = model_manager

    def get_default_model(self):
        """获取默认模型"""
        return self._model_manager.get_default_model()

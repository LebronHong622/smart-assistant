"""
LLM 适配器 - 实现模型端口
"""

from typing import Any
from domain.shared.ports.model_port import ModelPort
from infrastructure.external.model.openai_model_config import MODEL_CONFIGS, DEFAULT_MODEL_NAME


class LLMAdapter(ModelPort):
    """LLM 适配器实现"""

    def __init__(self):
        self.models = {}
        self._load_models_from_config()

    def add_model(self, model_name: str, model: Any):
        """添加模型到适配器"""
        self.models[model_name] = model

    def get_model(self, model_name: str = DEFAULT_MODEL_NAME) -> Any:
        """获取模型实例"""
        return self.models.get(model_name)

    def _load_models_from_config(self):
        """从配置文件加载模型"""
        for model_config_name, model_config in MODEL_CONFIGS.items():
            self.add_model(model_config_name, model_config)

    def get_default_model(self) -> Any:
        """获取默认模型实例"""
        return self.get_model(DEFAULT_MODEL_NAME)

"""
模型管理模块
"""
from typing import Any
from langchain_openai import ChatOpenAI
from config.settings import settings
from infrastructure.model.openai_model_config import MODEL_CONFIGS, DEFAULT_MODEL_NAME

class ModelManager:
    def __init__(self):
        self.models = {}
        # 初始化时从配置文件加载模型
        self._load_models_from_config()

    def add_model(self, model_name: str, model: Any):
        """添加模型到管理器"""
        self.models[model_name] = model

    def get_model(self, model_name: str = DEFAULT_MODEL_NAME) -> Any:
        """获取模型实例"""
        return self.models.get(model_name)

    def _load_models_from_config(self):
        """从配置文件加载模型"""
        for model_config_name, model_config in MODEL_CONFIGS.items():
            # 添加到管理器
            self.add_model(model_config_name, model_config)

    def get_default_model(self) -> Any:
        """获取默认模型实例"""
        return self.get_model(DEFAULT_MODEL_NAME)

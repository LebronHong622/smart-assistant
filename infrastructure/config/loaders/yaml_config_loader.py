"""
YAML配置加载器
支持环境变量占位符替换 ${VAR_NAME}
"""
import os
import re
from typing import Any, Dict
import yaml
from domain.eval.test_dataset_config import TestDatasetConfig


class YamlConfigLoader:
    """YAML配置加载器"""

    # 匹配 ${VAR_NAME} 格式的环境变量占位符
    _env_pattern = re.compile(r'\$\{([^}]+)\}')

    @classmethod
    def load_from_file(cls, config_path: str) -> TestDatasetConfig:
        """从YAML文件加载配置

        Args:
            config_path: YAML文件路径

        Returns:
            类型化的TestDatasetConfig对象
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        # 替换环境变量占位符
        substituted_data = cls._substitute_env_vars(raw_data)

        # 解析YAML
        config_dict = yaml.safe_load(substituted_data)

        # 转换为类型化配置对象
        return TestDatasetConfig(**config_dict)

    @classmethod
    def _substitute_env_vars(cls, content: str) -> str:
        """替换内容中的环境变量占位符

        Args:
            content: 原始YAML内容

        Returns:
            替换后的内容
        """
        def replace_match(match) -> str:
            env_name = match.group(1)
            env_value = os.getenv(env_name, "")
            return env_value

        return cls._env_pattern.sub(replace_match, content)

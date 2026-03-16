"""
YAML配置加载器实现
"""
import yaml
from pathlib import Path
from typing import Dict, List
from infrastructure.external.tool.loaders.config_loader_port import ToolConfigLoaderPort
from infrastructure.core.log import app_logger


class YamlConfigLoader(ToolConfigLoaderPort):
    """YAML格式工具配置加载器，内部管理配置文件路径"""

    def __init__(self, config_path: str = None):
        """
        构造函数，可指定配置文件路径，默认使用项目标准路径

        Args:
            config_path: 可选配置文件路径
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # 默认使用项目标准配置路径
            self.config_path = Path(__file__).parent.parent.parent.parent / "infrastructure" / "prompt" / "tool_config.yaml"

    def load_config(self) -> Dict[str, List[str]]:
        """加载YAML格式的工具配置"""
        try:
            if not self.config_path.exists():
                app_logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return {"default": ["get_weather", "document_retrieval"]}

            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            app_logger.info(f"YAML工具配置加载成功，支持的Agent类型: {list(config.keys())}")
            return config

        except Exception as e:
            app_logger.error(f"YAML工具配置加载失败: {str(e)}，使用默认配置")
            return {"default": ["get_weather", "document_retrieval"]}
import os
import yaml
from typing import Dict, Any
from infrastructure.core.log import app_logger
from .base_loader import BaseTemplateLoader


class YamlTemplateLoader(BaseTemplateLoader):
    """YAML template loader supporting multiple files and hot reload"""

    def __init__(self, template_dir: str = "config/prompt"):
        self.template_dir = self._resolve_template_dir(template_dir)
        self.last_modified: Dict[str, float] = {}
        app_logger.info(f"YAML template loader initialized with directory: {self.template_dir}")

    def _resolve_template_dir(self, template_dir: str) -> str:
        """Resolve absolute path to template directory"""
        if os.path.isabs(template_dir):
            return template_dir
        # Resolve relative to project root
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            template_dir
        )

    def load_templates(self) -> Dict[str, Any]:
        """Load all YAML templates from the directory"""
        templates: Dict[str, Any] = {}
        self.last_modified.clear()

        for filename in os.listdir(self.template_dir):
            if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                continue

            file_path = os.path.join(self.template_dir, filename)
            try:
                # 提取文件名（无后缀）作为前缀
                file_prefix = os.path.splitext(filename)[0]

                with open(file_path, 'r', encoding='utf-8') as f:
                    file_templates = yaml.safe_load(f) or {}
                    # 给每个 key 添加文件名前缀
                    prefixed_templates = {
                        f"{file_prefix}.{key}": value
                        for key, value in file_templates.items()
                    }
                    templates.update(prefixed_templates)

                self.last_modified[file_path] = os.path.getmtime(file_path)
                app_logger.debug(f"Loaded {len(file_templates)} templates from {filename}")

            except Exception as e:
                app_logger.error(f"Failed to load template file {filename}: {str(e)}")

        app_logger.info(f"Total loaded templates: {len(templates)}")
        app_logger.info(f"Total loaded templates: {templates.get('default')}")
        return templates

    def is_modified(self) -> bool:
        """Check if any template file has been modified"""
        for file_path, last_mtime in self.last_modified.items():
            if not os.path.exists(file_path):
                return True
            current_mtime = os.path.getmtime(file_path)
            if current_mtime != last_mtime:
                return True
        return False

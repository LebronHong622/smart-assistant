"""
提示词管理模块
"""
import os
import yaml
from infrastructure.log.log import app_logger

PROMPT_TEMPLATE_FILE = "general_qa.yaml"

DEFAULT_PROMPT = """你是一个友好的助手。用户说: {query}
        请简洁友好地回答用户的问题。
"""


class PromptManager:
    def __init__(self, prompt_template_file: str = PROMPT_TEMPLATE_FILE):
        self.prompt_template_file = os.path.join(
            os.path.dirname(__file__),  prompt_template_file
        )
        self.templates = {}
        self.last_modified_time = 0  # 记录文件最后修改时间
        self._load_templates()

    def _load_templates(self):
        """加载提示词模板文件"""
        try:
            # 读取YAML文件内容
            with open(self.prompt_template_file, 'r', encoding='utf-8') as f:
                self.templates = yaml.safe_load(f) or {}
            # 更新文件最后修改时间
            if os.path.exists(self.prompt_template_file):
                self.last_modified_time = os.path.getmtime(self.prompt_template_file)
            app_logger.debug(f"提示词模板文件加载成功: {self.prompt_template_file}")
        except FileNotFoundError:
            app_logger.error(f"提示词模板文件不存在: {self.prompt_template_file}")
        except yaml.YAMLError as e:
            app_logger.error(f"解析YAML文件失败: {e}")
        except Exception as e:
            app_logger.error(f"加载提示词模板失败: {e}")

    def load_prompt_templates(self, template_type: str = "default") -> str:
        """加载提示词模板

        Args:
            template_type: 模板类型

        Returns:
            对应的提示词模板字符串
        """
        # 检查文件是否被修改，如果是则重新加载
        self._check_and_reload_templates()

        # 优先从加载的模板中获取
        if template_type in self.templates:
            return self.templates[template_type]["content"]

        # 如果指定类型的模板不存在，返回默认模板
        app_logger.warning(f"未找到模板类型: {template_type}，使用默认模板")
        return DEFAULT_PROMPT

    def _check_and_reload_templates(self):
        """检查模板文件是否被修改，如果是则重新加载"""
        try:
            if os.path.exists(self.prompt_template_file):
                current_modified_time = os.path.getmtime(self.prompt_template_file)
                if current_modified_time != self.last_modified_time:
                    app_logger.info(f"检测到提示词模板文件被修改，重新加载: {self.prompt_template_file}")
                    self._load_templates()
        except Exception as e:
            app_logger.error(f"检查模板文件修改时间失败: {e}")

    def refresh_templates(self):
        """手动刷新模板文件（强制重新加载）"""
        app_logger.info(f"手动刷新提示词模板文件: {self.prompt_template_file}")
        self._load_templates()

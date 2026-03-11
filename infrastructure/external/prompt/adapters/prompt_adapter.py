"""
Prompt adapter implementing PromptPort interface
"""
from typing import Any, Dict, Optional
from domain.shared.ports.prompt_port import PromptPort
from infrastructure.external.prompt.prompt_manager import PromptManager
from infrastructure.core.log.log import app_logger


class PromptAdapter(PromptPort):
    """Prompt management adapter implementation"""

    def __init__(self, prompt_template_file: str = "general_qa.yaml"):
        self.prompt_manager = PromptManager(prompt_template_file)
        app_logger.debug("PromptAdapter initialized")

    def get_prompt(self, prompt_key: str, **kwargs: Any) -> str:
        """
        Get prompt template by key and render with parameters

        Args:
            prompt_key: Unique key of the prompt template
            **kwargs: Parameters to render the template

        Returns:
            Rendered prompt string
        """
        template_content = self.prompt_manager.load_prompt_templates(prompt_key)
        try:
            rendered = template_content.format(**kwargs)
            app_logger.debug(f"Prompt rendered for key: {prompt_key}")
            return rendered
        except KeyError as e:
            app_logger.error(f"Missing parameter {e} when rendering prompt {prompt_key}")
            return template_content

    def load_prompts(self) -> None:
        """Load/reload all prompt templates from storage"""
        self.prompt_manager.refresh_templates()
        app_logger.info("Prompt templates reloaded")

    def get_template(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """
        Get raw prompt template without rendering

        Args:
            prompt_key: Unique key of the prompt template

        Returns:
            Template dictionary if exists, None otherwise
        """
        # Check and reload templates if modified
        self.prompt_manager._check_and_reload_templates()
        return self.prompt_manager.templates.get(prompt_key)

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar
from domain.shared.ports.prompt_port import PromptPort
from infrastructure.external.prompt.loaders.base_loader import BaseTemplateLoader
from infrastructure.core.log import app_logger

T = TypeVar('T')  # 泛型类型，由具体框架实现决定


class BaseFrameworkPrompt(ABC, PromptPort[T]):
    """Base class for framework-specific prompt implementations"""

    def __init__(self, template_loader: BaseTemplateLoader):
        self.template_loader = template_loader
        self._templates: Dict[str, Any] = {}
        self.load_prompts()
        app_logger.debug(f"{self.__class__.__name__} initialized")

    def load_prompts(self) -> None:
        """Load/reload all templates"""
        self._templates = self.template_loader.load_templates()

    def get_template(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """Get raw template by key, auto-reload if modified"""
        if self.template_loader.is_modified():
            app_logger.info("Templates modified, reloading...")
            self.load_prompts()
        return self._templates.get(prompt_key)

    def get_prompt_string(self, prompt_key: str, **kwargs: Any) -> str:
        """Get rendered prompt as string (backward compatibility)"""
        template = self.get_template(prompt_key)
        if not template:
            raise ValueError(f"Prompt template not found: {prompt_key}")

        content = template.get("content", "")
        try:
            return content.format(**kwargs)
        except KeyError as e:
            app_logger.error(f"Missing parameter {e} when rendering prompt {prompt_key}")
            return content

    @abstractmethod
    def get_prompt(self, prompt_key: str, **kwargs: Any) -> T:
        """Get framework-native prompt object"""
        pass

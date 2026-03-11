from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PromptPort(ABC):
    """Prompt management port interface"""

    @abstractmethod
    def get_prompt(self, prompt_key: str, **kwargs: Any) -> str:
        """
        Get prompt template by key and render with parameters

        Args:
            prompt_key: Unique key of the prompt template
            **kwargs: Parameters to render the template

        Returns:
            Rendered prompt string
        """
        pass

    @abstractmethod
    def load_prompts(self) -> None:
        """Load/reload all prompt templates from storage"""
        pass

    @abstractmethod
    def get_template(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """
        Get raw prompt template without rendering

        Args:
            prompt_key: Unique key of the prompt template

        Returns:
            Template dictionary if exists, None otherwise
        """
        pass

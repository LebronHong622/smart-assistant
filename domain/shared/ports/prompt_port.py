from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic

T = TypeVar('T')  # 泛型类型，由具体框架实现决定


class PromptPort(ABC, Generic[T]):
    """Prompt management port interface"""

    @abstractmethod
    def get_prompt(self, prompt_key: str, **kwargs: Any) -> T:
        """
        Get framework-native prompt object by key

        Args:
            prompt_key: Unique key of the prompt template
            **kwargs: Parameters for rendering (if needed)

        Returns:
            Framework-specific prompt object ready for use
        """
        pass

    @abstractmethod
    def load_prompts(self) -> None:
        """Load/reload all prompt templates from storage"""
        pass

    @abstractmethod
    def get_template(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """
        Get raw template metadata without rendering

        Returns:
            Template dictionary if exists, None otherwise
        """
        pass

    @abstractmethod
    def get_prompt_string(self, prompt_key: str, **kwargs: Any) -> str:
        """
        Get rendered prompt as plain string (for backward compatibility)

        Returns:
            Rendered prompt string
        """
        pass

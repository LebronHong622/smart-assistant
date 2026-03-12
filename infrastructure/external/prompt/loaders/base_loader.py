from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTemplateLoader(ABC):
    """Base interface for template loaders"""

    @abstractmethod
    def load_templates(self) -> Dict[str, Any]:
        """
        Load all templates from storage
        Returns:
            Dictionary of templates, key is prompt_key, value is template data
        """
        pass

    @abstractmethod
    def is_modified(self) -> bool:
        """
        Check if templates have been modified since last load
        Returns:
            True if modified, False otherwise
        """
        pass

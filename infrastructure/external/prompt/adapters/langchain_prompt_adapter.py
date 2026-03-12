from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from infrastructure.external.prompt.adapters.base_framework_prompt import BaseFrameworkPrompt
from infrastructure.core.log import app_logger


class LangChainPromptAdapter(BaseFrameworkPrompt[PromptValue]):
    """LangChain-specific prompt adapter returning ChatPromptValue for direct LLM invocation"""

    def get_prompt(self, prompt_key: str, **kwargs: Any) -> PromptValue:
        """
        Get ChatPromptValue ready for direct LLM invoke()
        Renders the template with provided kwargs and returns ChatPromptValue
        
        Args:
            prompt_key: Unique key of the prompt template
            **kwargs: Parameters for rendering the template
            
        Returns:
            ChatPromptValue ready for langchain model invoke()
        """
        template = self.get_template(prompt_key)
        if not template:
            raise ValueError(f"Prompt template not found: {prompt_key}")

        # Support structured templates with system/human/ai roles
        if isinstance(template, dict) and any(key in template for key in ["system", "human", "ai"]):
            messages = []
            if "system" in template:
                messages.append(("system", template["system"]))
            if "human" in template:
                messages.append(("human", template["human"]))
            if "ai" in template:
                messages.append(("ai", template["ai"]))
            prompt_template = ChatPromptTemplate.from_messages(messages)
            return prompt_template.invoke(kwargs)

        # Fallback to simple content field (original yaml format)
        content = template.get("content", "")
        prompt_template = ChatPromptTemplate.from_messages([("human", content)])
        return prompt_template.invoke(kwargs)

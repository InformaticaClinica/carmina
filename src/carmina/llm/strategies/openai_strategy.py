import openai
from typing import List, Dict

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.utils.token_counter import get_token_counter


class OpenAIStrategy(BaseLLMStrategy):
    """
    Implementation for OpenAI models.
    """

    # Dictionary to map model names to their context windows
    _context_windows = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 8192,
        "gpt-4-turbo-32k": 32768,
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.token_counter = get_token_counter(self.model_name, "openai")

    def run_inference(self, messages, inference_params):
        inference_params = {
            k: v
            for k, v in inference_params.items()
            if v is not None and (not hasattr(v, "__len__") or len(v) > 0)
        }
        messages = self.adapt_message(messages)
        if self.provider_name == "openai" or self.provider_name == "local":
            return self.cloud_provider.run_inference(
                model_id=self.model_name, messages=messages, **inference_params
            )

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        pass

    def get_context_window(self) -> int:
        """
        Get the maximum context window size for this model.

        Returns:
            Maximum number of tokens the model can process
        """
        # First check model_config.py
        model_name_lower = self.model_name.lower()
        if model_name_lower in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name_lower]["context_window"]
        return self._context_windows.get(self.model_name, 4096)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using OpenAI tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        return self.token_counter.count_tokens(text)

    def adapt_message(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Adapts the message format for OpenAI.

        Args:
            messages: List of messages to adapt

        Returns:
            List of adapted messages
        """
        adapted_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                adapted_messages.append({"role": "developer", "content": content})
            elif role == "user":
                adapted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                adapted_messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Unknown role: {role}")
        return adapted_messages

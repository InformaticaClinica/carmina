import openai
import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.model_config import MODEL_CONFIGS

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
    
    def run_inference(self, messages, inference_params):
        inference_params = {k: v for k, v in inference_params.items() if v is not None and (not hasattr(v, '__len__') or len(v) > 0)}
        messages = self.adapt_message(messages)
        if self.provider_name == "openai":
            return self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                **inference_params
            )

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        pass

    def get_context_window(self) -> int:
        pass

    def count_tokens(self, text: str) -> int:
        pass

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


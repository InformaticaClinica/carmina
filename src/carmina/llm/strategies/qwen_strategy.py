import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.model_config import MODEL_CONFIGS


class QwenStrategy(BaseLLMStrategy):
    """
    Implementation for Qwen models.
    """

    # Dictionary to map model names to their context windows
    _context_windows = {
        "qwen2.5-72b-instruct": 131072,
        "qwen2.5-32b-instruct": 131072,
        "qwen2.5-14b-instruct": 131072,
        "qwen2.5-7b-instruct": 131072,
        "qwen2.5-3b-instruct": 32768,
        "qwen2.5-1.5b-instruct": 32768,
        "qwen2.5-0.5b-instruct": 32768,
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.anonymization_mode = os.environ.get("ANONYMIZATION_MODE") or kwargs.get("anonymization_mode", "label")
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get("temperature", 0.7)
        self.max_tokens = os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens", 2500)
        self.top_p = os.environ.get("TOP_P") or kwargs.get("top_p", 1.0)
        self.frequency_penalty = os.environ.get("FREQUENCY_PENALTY") or kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = os.environ.get("PRESENCE_PENALTY") or kwargs.get("presence_penalty", 0.0)
        self.model_name = model_name
        self.cloud_provider = cloud_provider
        self.provider_name = self.cloud_provider.get_name()
    
    def identify(self, text: str, **kwargs) -> str:
        #TODO: Adapter design pattern for Qwen
        system_prompt = load_system_prompt("identify")
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        if self.provider_name == "local":
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty
                }
            )
            return self.adapt_respose(response)
    
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
        pass

    def process_for_anonymization(self, text: str, strategy: str) -> Dict[str, Any]:
        pass
    
    def adapt_respose(self, response):
        idx = response.find("</think>")
        if idx != -1:
            # qwen adds a few newlines so we remove them as well
            response = response[idx + 8:].lstrip()
        return response

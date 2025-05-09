import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.model_config import MODEL_CONFIGS

class LlamaStrategy(BaseLLMStrategy):
    """
    Implementation for Llama models.
    """
    # Dictionary to map model names to their context windows
    _context_windows = {
        "llama-3.2-1b": 8192,
        "llama-3.2-3b": 8192,
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
    
    def identify(self, text: str, **kwargs) -> str:
        system_prompt = load_system_prompt("identify")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        if self.provider_name == "local" or self.provider_name == "mock":
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
            # LocalProvider already extracts the message content, so just return it
            return response
        else:
            raise ValueError(f"Unsupported {self.provider_name} provider type for identification")

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        pass

    def get_context_window(self) -> int:
        return self._context_windows.get(self.model_name, 4096)

    def count_tokens(self, text: str) -> int:
        pass

    def process_for_anonymization(self, text: str, strategy: str) -> Dict[str, Any]:
        pass
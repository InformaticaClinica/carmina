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
        print("here")
        #TODO: Adapter design pattern for OpenAI
        system_prompt = load_system_prompt("identify")
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        if self.provider_name == "openai":
            self.cloud_provider.run_inference(
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
            
        pass

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        pass

    def get_context_window(self) -> int:
        pass

    def count_tokens(self, text: str) -> int:
        pass

    def process_for_anonymization(self, text: str, strategy: str) -> Dict[str, Any]:
        pass

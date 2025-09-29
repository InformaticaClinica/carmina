import os
from typing import List, Dict, Any
import logging 
logger = logging.getLogger(__name__)
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
        "llama-3.3-70b": 131072,
        "llama-4-scout":131072,

    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
    
    def identify(self, text: str, **kwargs) -> str:
        system_prompt = load_system_prompt("identify")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        if self.provider_name in ["aws", "mock", "local", "mock_ollama"]:
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
            if self.provider_name=="aws":
                return response.get("generation",).strip()
            # LocalProvider already extracts the message content, so just return it
            return response
        else:
            raise ValueError(f"Unsupported {self.provider_name} provider type for identification")

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process multiple texts in batch for identification.
        """
        results = []
        for text in texts:
            result = self.identify(text, **kwargs)
            results.append(result)
        return results

    def get_context_window(self) -> int:
        return self._context_windows.get(self.model_name, 4096)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        For Llama models, this is an approximation.
        """
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def run_inference(self, messages, inference_params) -> str:
        """
        Run inference using the configured cloud provider.
        """
        if self.provider_name in ["aws", "mock", "local", "mock_ollama"]:
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params=inference_params
            )
            if self.provider_name == "aws":
                return response.get('generation', {})
            return response
        else:
            raise ValueError(f"Unsupported {self.provider_name} provider type for LlamaStrategy")

    def process_for_anonymization(self, text: str, strategy: str) -> Dict[str, Any]:
        """
        Process text for anonymization using the specified strategy.
        """
        messages = self.get_message(strategy, text)
        inference_params = self.get_inference_params()
        result = self.run_inference(messages, inference_params)
        return result
import os
from typing import List, Dict, Any
import logging 
logger = logging.getLogger(__name__)
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy

class MistralStrategy(BaseLLMStrategy):
    """
    Implementation for Mistral models.
    """
    _context_windows = {
        "mistral.mistral-7b-instruct-v0:2": 8192,
        "mistral.mistral-large-2402-v1:0": 32768
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)

    def identify(self, text: str, **kwargs) -> str:
        messages = self.get_message("identify", text)
        inference_params = self.get_inference_params()
        response = self.run_inference(messages, inference_params)
        return response

    def process_for_anonymization(self, text: str, strategy: str) -> str:
        messages = self.get_message(strategy, text)
        inference_params = self.get_inference_params()
        response = self.run_inference(messages, inference_params)
        return response

    def run_inference(self, messages, inference_params) -> str:
        if self.provider_name in ["aws", "mock", "local", "mock_ollama"]:
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params=inference_params
            )
            if "large" in self.model_name.lower():
                return response["choices"][0]["message"]["content"]
            else:
                return response["outputs"][0]["text"]
        
        else:
            raise ValueError(f"Unsupported provider {self.provider_name} for MistralStrategy")

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        return [self.identify(text, **kwargs) for text in texts]

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_context_window(self) -> int:
        return self._context_windows.get(self.model_name, 8192)

import os
import logging 
logger = logging.getLogger(__name__)

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.model_config import MODEL_CONFIGS
from src.carmina.llm.utils.prompt_loader import load_system_prompt

class DeepSeekStrategy(BaseLLMStrategy):
    """Implementation for DeepSeek models."""

    # Dictionary to map model names to their context windows
    _context_windows = {
        "deepseek-r1": 200000,
        "deepseek-v3": 180000,
        "deepseek-v3-0324": 150000,
    }

    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
    
    def run_inference(self, messages, inference_params) -> str:
        inference_params = {k: v for k, v in inference_params.items() if v is not None and (not hasattr(v, '__len__') or len(v) > 0)}
        if self.provider_name == "aws":
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params=inference_params
            )
            response=response["choices"][0]["message"]["content"]
            return self.adapt_respose(response)
        if self.provider_name == "local":
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params=inference_params
            )
            return self.adapt_respose(response)
        elif self.provider_name == "azure" or self.provider_name == "mock":
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                **inference_params
            )
            return self.adapt_respose(response)
        
        else:
            raise NotImplementedError(f"Provider {self.provider_name} not implemented for DeepSeek.")
    
    def process_for_anonymization(self, text: str, strategy: str) -> str:
        """
        Process text for anonymization using the specified strategy.
        
        Args:
            text: The input text to anonymize
            strategy: The anonymization strategy to use (e.g., 'label')
            
        Returns:
            The anonymized text
        """
        messages = self.get_message(strategy, text)
        inference_params = self.get_inference_params()
        result = self.run_inference(messages, inference_params)
        return result
    
    def batch_identify(self, texts, **kwargs):
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
        return None
    
    def count_tokens(self, text):
        pass

    def process_for_identification(self, text, mode):
        pass

    def adapt_respose(self, response):
        if "deepseek-r1" in self.model_name:
            idx = response.find("</think>")
            if idx != -1:
                # deepseek adds a few newlines so we remove them as well
                response = response[idx + 8:].lstrip()
            return response.strip()
        elif self.model_name == "deepseek-v3":
            return response
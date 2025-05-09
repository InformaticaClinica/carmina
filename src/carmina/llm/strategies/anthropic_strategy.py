import os
import logging

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.model_config import MODEL_CONFIGS
from src.carmina.llm.utils.prompt_loader import load_system_prompt

class AnthropicStrategy(BaseLLMStrategy):
    """Implementation for Anthropic Claude models."""
    
    # Dictionary to map model names to their context windows
    _context_windows = {
        "claude-3.5-sonnet": 200000,
        "claude-3.7-sonnet": 220000,
    }
    
    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
    
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
        return None # Default if model not found
    

    def identify(self, text: str, **kwargs) -> str:
        """
        Identify sensitive information in the input text.
        
        Args:
            text: Input text to process
            **kwargs: Additional identification parameters
        Returns:
            Text with identified sensitive information
        """
        system_prompt = load_system_prompt("identify")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        inference_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            }
        if self.provider_name == "aws":
            input_data = {"messages": messages}
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=input_data,
                inference_params=inference_params
            )
            return response["content"][0]["text"]

        if self.provider_name == "mock":
            return self.cloud_provider.run_inference(
                model_id = self.model_name,
                messages = messages,
                inference_params = inference_params
            )
            

    def batch_identify(self, texts, **kwargs):
        pass

    
    def count_tokens(self, text):
        pass


    def process_for_anonymization(self, text, mode):
        pass

    def process_for_anonymization(self, text, mode):
        pass
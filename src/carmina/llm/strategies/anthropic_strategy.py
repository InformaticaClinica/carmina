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
    
    def run_inference(self, messages, inference_params) -> str:
        """
        Run inference using the specified model and parameters.

        Args:
            messages: List of messages to send to the model
            inference_params: Dictionary of inference parameters
        Returns:
            Model's response as a string
        """
        # Remove empty parameters from inference_params
        inference_params = {k: v for k, v in inference_params.items() if v is not None and (not hasattr(v, '__len__') or len(v) > 0)}
        
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

    def identify(self, text: str, **kwargs) -> str:
        """
        Identify sensitive information in the input text.
        
        Args:
            text: Input text to process
            **kwargs: Additional identification parameters
        Returns:
            Text with identified sensitive information
        """
        messages = self.get_message("identify", text)
        inference_params = self.get_inference_params()
        return self.run_inference(messages, inference_params)
            

    def batch_identify(self, texts, **kwargs):
        pass

    
    def count_tokens(self, text):
        pass

    def process_for_anonymization(self, text, mode):
        """
        Process the text for anonymization based on the specified mode.
        
        Args:
            text: Input text to process
            mode: Mode of processing (e.g., "label", "substitution")
        Returns:
            Processed text
        """
        messages = self.get_message(mode, text)
        inference_params = self.get_inference_params()
        return self.run_inference(messages, inference_params)

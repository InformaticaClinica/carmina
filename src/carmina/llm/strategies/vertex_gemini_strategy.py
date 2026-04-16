import os

from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.model_config import MODEL_CONFIGS
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.utils.token_counter import get_token_counter

class VertexGeminiStrategy(BaseLLMStrategy):
    """
    Implementation for Gemini models on Vertex AI.
    """
    # Dictionary to map model names to their context windows
    #TODO: CHECK CONTEXT WINDOWS
    _context_windows = {
        "gemini-2.5-flash": 1000000,  # Vertex AI supports larger contexts
        "gemini-2.5-pro": 2000000,
        "gemini-1.5-flash": 1000000,
        "gemini-1.5-pro": 2000000,
    }
    
    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.token_counter = get_token_counter(self.model_name, self.provider_name)
    
    
    def run_inference(self, messages, inference_params):
        """
        Run inference on the model with the provided messages and parameters.
        
        Args:
            messages (list): List of messages to send to the model.
            inference_params (dict): Inference parameters for the model.
        
        Returns:
            str: The model's response.
        """
        # Filter None and empty values from inference_params
        inference_params = {k: v for k, v in inference_params.items() if v is not None and (not hasattr(v, '__len__') or len(v) > 0)}
        
        # Use inference_params if provided, otherwise fall back to strategy defaults
        params_to_use = {
            "temperature": inference_params.get("temperature", self.temperature),
            "max_tokens": inference_params.get("max_tokens", self.max_tokens),
            "top_p": inference_params.get("top_p", self.top_p)
        }
        
        response = self.cloud_provider.run_inference(
            model_id=self.model_name,
            messages=messages,
            **params_to_use
        )
        return response
    
    def identify(self, text, **kwargs):
        message = self.get_message("identify", text) #preguntar
        inference_params = self.get_inference_params()
        return self.run_inference(
            messages=message,
            inference_params=inference_params
        )
        
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
        return self._context_windows.get(self.model_name, 1000000)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using Gemini tokenizer approximation.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        return self.token_counter.count_tokens(text)
    
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


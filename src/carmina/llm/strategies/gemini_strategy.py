import os

from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.model_config import MODEL_CONFIGS
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider

class GeminiStrategy(BaseLLMStrategy):
    """
    Implementation for Gemini models.
    """
    # Dictionary to map model names to their context windows
    #TODO: CHECK CONTEXT WINDOWS
    _context_windows = {
        "gemini-2.5-flash": 200000,
        "gemini-2.5-pro": 180000,
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)

    def run_inference(self, messages, inference_params):
        """
        Run inference on the model with the provided messages and parameters.
        
        Args:
            messages (list): List of messages to send to the model.
            inference_params (dict): Inference parameters for the model.
        
        Returns:
            str: The model's response.
        """
        inference_params = {k: v for k, v in inference_params.items() if v is not None and (not hasattr(v, '__len__') or len(v) > 0)}
        if self.provider_name == "aws":
            raise NotImplementedError(f"Provider {self.provider_name} not implemented for Gemini.")
        elif self.provider_name == "azure":
            raise NotImplementedError(f"Provider {self.provider_name} not implemented for Gemini.")
        elif self.provider_name == "google_ai_studio":
            inference_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                **inference_params
            )
            return response
        elif self.provider_name == "local":
            if "gemma" not in self.model_name.lower():
                raise ValueError(f"Provider {self.provider_name} not supported for Gemini.")
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
            raise ValueError(f"Provider {self.provider_name} not supported for Gemini.")

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
        return None
    
    def count_tokens(self, text):
        pass

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

    def process_for_identification(self, text, mode):
        pass

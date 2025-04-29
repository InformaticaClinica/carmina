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
        self.provider_name = self.cloud_provider.get_name()
        self.anonymization_mode = os.environ.get("ANONYMIZATION_MODE") or kwargs.get("anonymization_mode", "label")
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get("temperature", 0.7)
        self.max_tokens = os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens", 2500)
        self.top_p = os.environ.get("TOP_P") or kwargs.get("top_p", 1.0)

    def identify(self, text, **kwargs):
        system_prompt = load_system_prompt("identify")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
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
        else:
            raise ValueError(f"Provider {self.provider_name} not supported for Gemini.")
        
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

    def process_for_anonymization(self, text, mode):
        pass

    def process_for_identification(self, text, mode):
        pass

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
        self.provider_name = self.cloud_provider.get_name()
        self.anonymization_mode =  os.environ.get("ANONYMIZATION_MODE") or kwargs.get("anonymization_mode", "label")
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get("temperature", 0.7)
        self.max_tokens = os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens", 1000)
        self.top_p = os.environ.get("TOP_P") or kwargs.get("top_p", 1.0)
        self.frequency_penalty = os.environ.get("FREQUENCY_PENALTY") or kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = os.environ.get("PRESENCE_PENALTY") or kwargs.get("presence_penalty", 0.0)

        
    
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
        if self.provider_name == "aws":
            # Use AWS Bedrock through cloud provider
            input_data = {"messages": messages}
            inference_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Call the cloud provider's run_inference method
            response = self.cloud_provider.run_inference(
                model_name=self.model_name,
                messages=input_data,
                inference_params=inference_params
            )
            
            # Extract content from AWS Bedrock response format
            if "content" in response and isinstance(response["content"], list):
                return response["content"][0]["text"]
            else:
                logging.warning(f"Unexpected response format from AWS: {response}")
                return response.get("completion", text)

    def batch_identify(self, texts, **kwargs):
        pass

    
    def count_tokens(self, text):
        pass


    def process_for_anonymization(self, text, mode):
        pass

    def process_for_anonymization(self, text, mode):
        pass
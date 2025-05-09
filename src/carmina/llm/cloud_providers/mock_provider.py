from typing import Dict, Any, List, Optional
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider

class MockProvider(BaseCloudProvider):
    def __init__(self, name: str = "mock", **kwargs):
        super().__init__(name, **kwargs)

    def get_model(self):
        return f"Mock model from {self.name}"

    def get_tokenizer(self):
        return f"Mock tokenizer from {self.name}"

    def generate_text(self, prompt: str):
        return f"Generated text for '{prompt}' using {self.name}"
    
    def run_inference(self, model_id: str, messages: List[Dict[str, str]], **kwargs):
        """
        Mock inference method that simulates running a model.
        
        Args:
            model_id (str): The model ID to use (e.g., "llama-3.2-3b").
            messages (List[Dict[str, str]]): The input messages for the model.
            **kwargs: Additional arguments for the inference.
        
        Returns:
            str: Simulated response from the model.
        """
        return messages[1]["content"]
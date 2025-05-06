import openai
from openai import OpenAI
import os
import logging

from typing import List, Dict, Any
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider

class OpenAIProvider(BaseCloudProvider):
    """
    OpenAI cloud provider implementation.
    """
    _openai_model_ids = {
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-o4-mini": "o4-mini-2025-04-16",
        "gpt-o3": "o3-2025-04-16"
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the OpenAI provider with API key and organization.
        """
        self.api_key = os.environ.get("OPENAI_API_KEY") or kwargs.get("api_key", None)
        openai.api_key = self.api_key
        self._client = OpenAI()
    
    def run_inference(self, model_id: str, messages: dict, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.get_model_id(model_id),
            messages=messages,
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
        )
        return response.choices[0].message

    def get_model_id(self, model_name: str) -> str:
        """
        Returns the model ID for the given model name.
        
        Args:
            model_name (str): The name of the model.
        
        Returns:
            str: The corresponding model ID.
        """
        return self._openai_model_ids.get(model_name, model_name)
    
    def get_name(self) -> str:
        """
        Returns the name of the cloud provider.
        
        Returns:
            str: Cloud provider name.
        """
        return "openai"

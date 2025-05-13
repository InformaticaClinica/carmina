# -*- coding: utf-8 -*-
import os
from azure.ai.inference import ChatCompletionsClient

from azure.ai.inference.models import ChatRequestMessage
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from .base_provider import BaseCloudProvider
import logging

class AzureProvider(BaseCloudProvider):
    """
    Azure provider for accessing LLMs through Microsoft Azure services.
    This provider enables access to models like DeepSeek through Azure.
    """
    _azure_model_ids = {
        # Deepseek models
        "deepseek-r1": "DeepSeek-R1",
        "deepseek-v3":"DeepSeek-V3",
        "deepseek-v3-0324": "DeepSeek-V3-0324",
        # UPCOMING: Add more models as needed
        # "model-name": "azure.model-name:version",
        }

    def __init__(self, name:str="azure", **kwargs):
        super().__init__(name="azure", **kwargs)
        self.endpoint = os.environ.get("AZURE_ENDPOINT") or kwargs.get("endpoint", None)
        self.api_key = os.environ.get("AZURE_SUBSCRIPTION_ID") or kwargs.get("azure_subscription_id", None)
        if self.api_key is None:
            raise ValueError("API key is required for Azure provider.")
        if self.endpoint is None:
            raise ValueError("Deployment URL is required for Azure provider.")
        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version="2024-05-01-preview"
        )
        logging.info("Azure Provider initialized.")

    def connect(self) -> str:
        """
        Establish a connection to Azure services.
        Returns:
            str: Connection status message.
        """
        try:
            # Simple validation - check if client is initialized
            # and if credentials are available
            if not self._client or not self.api_key or not self.endpoint:
                raise ValueError("Azure client not properly initialized")
            
            # No need to make an actual API call here
            # We just verify the client is set up correctly
            self._is_initialized = True
            logging.info("Azure credentials verified")
            return "Connected to Azure API"
        except Exception as e:
            logging.error(f"Failed to connect to Azure API: {e}")
            raise

    def run_inference(self, model_id: str, messages: dict, **kwargs) -> str:
        """
        Ejecuta la inferencia usando un modelo en Azure.
        Esto dependerá de cómo tengas desplegado DeepSeek en Azure
        (ej. Azure ML endpoint, VM, etc.).
        """
        logging.info(f"Running inference on Azure with model {model_id}")
        deployment_name = self._azure_model_ids.get(model_id)

        # Prepare parameters, only including non-None values
        params = {
            "messages": self.adapt_message(messages),
            "model": deployment_name,
        }

        # Add optional parameters only if they're not None
        if kwargs.get("max_tokens") is not None:
            params["max_tokens"] = kwargs.get("max_tokens")
        if kwargs.get("temperature") is not None:
            params["temperature"] = kwargs.get("temperature")
        if kwargs.get("top_p") is not None:
            params["top_p"] = kwargs.get("top_p")
        if kwargs.get("frequency_penalty") is not None:
            params["frequency_penalty"] = kwargs.get("frequency_penalty")
        if kwargs.get("presence_penalty") is not None:
            params["presence_penalty"] = kwargs.get("presence_penalty")

        # Call complete with the filtered parameters
        response = self._client.complete(**params)
        return response.choices[0].message.content

    def get_name(self) -> str:
        """
        Returns the name of the cloud provider.
        Returns:
            str: Cloud provider name.
        """
        return "azure"
    
    def adapt_message(self, messages: list) -> list:
        """
        Adapts the message format for Azure.
        Args:
            messages: List of messages to adapt
        Returns:
            List of adapted messages
        """
        adapted_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                adapted_messages.append(UserMessage(content=content))
            elif role == "assistant":
                adapted_messages.append(AssistantMessage(content=content))
            else:
                adapted_messages.append(SystemMessage(content=content))
        return adapted_messages
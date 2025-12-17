import os
import logging

from google import genai
from google.genai import types
from typing import List, Dict, Optional
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider


class GoogleAIStudioProvider(BaseCloudProvider):
    """
    Google AI Studio provider for running inference on models.

    This class is designed to interact with Google AI Studio's API for running inference
    on various models. It provides methods to initialize the provider, run inference, and
    manage model IDs.
    """

    _google_model_ids = {
        # Example model IDs
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-03-25",
        "gemini-3-pro": "gemini-3-pro-preview",
        "gemini-2.0-flash": "gemini-2.0-flash",
        # Add more models as needed
    }

    def __init__(self, **kwargs):
        self.api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY") or kwargs.get(
            "api_key", None
        )
        if self.api_key is None:
            raise ValueError("API key is required for Google AI Studio provider.")

        self._client = genai.Client(api_key=self.api_key)
        logging.info("Google AI Studio Provider initialized.")

    def run_inference(self, model_id: str, messages: List[Dict[str, str]], **kwargs):
        """
        Runs inference using a model in Google AI Studio.

        Args:
            model_id (str): The ID of the model to use for inference.
            messages (dict): The input messages for the model.
            **kwargs: Additional parameters for the inference request.

        Returns:
            str: The response from the model.
        """
        logging.info(f"Running inference on Google AI Studio with model {model_id}")
        model_name = self._google_model_ids.get(model_id)

        if not model_name:
            raise ValueError(f"Model ID {model_id} is not supported.")

        # Adapt the messages to the required format
        payload = self.adapt_message(messages, **kwargs)
        response = self._client.models.generate_content(
            model=model_name, contents=payload["contents"], config=payload["config"]
        )
        return response.text

    def get_name(self) -> str:
        """
        Returns the name of the cloud provider.

        Returns:
            str: Cloud provider name.
        """
        return "google_ai_studio"

    def adapt_message(self, messages: list, **kwargs) -> list:
        """
        Adapts the message format for Google AI Studio.

        Args:
            messages (list): List of messages to adapt.

        Returns:
            list: List of adapted messages.
        """
        payload = {}
        # Extract system instruction and user messages
        system_instruction, payload["contents"] = self.get_messages(messages)
        # Generate the payload for the inference request
        payload["config"] = self.generate_payload(system_instruction, **kwargs)
        return payload

    def get_messages(self, messages: list) -> list:
        """
        Returns the messages in the required format for Google AI Studio.

        Args:
            messages (list): List of messages to format.

        Returns:
            list: List of formatted messages.
        """
        formatted_messages = []
        system_instruction = None
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                formatted_messages.append(content)
            if role == "assistant":
                raise ValueError("Assistant role is not supported in Google AI Studio.")
        messages = self.check_user_message(formatted_messages)
        return system_instruction, messages

    def generate_payload(self, system_instruction: Optional[str], **kwargs) -> dict:
        """
        Generates the payload for the inference request.

        Args:
            system_instruction (Optional[str]): System instruction for the model, can be string or None.
            contents (list): The contents to include in the payload.
            **kwargs: Additional configuration parameters.

        Returns:
            dict: The generated payload.
        """
        # Check for system instruction
        if system_instruction:
            return types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
            )
        else:
            return types.GenerateContentConfig(
                max_output_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                top_k=kwargs.get("top_k"),
            )

    def check_user_message(self, messages: list) -> bool:
        if len(messages) > 1:
            raise ValueError(
                "Multiple user messages are not supported in Google AI Studio."
            )
        elif len(messages) == 0:
            raise ValueError("No user message provided.")
        elif len(messages) == 1:
            return messages[0]


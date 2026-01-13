"""
Base LLM strategy interface.

This module defines the abstract base class that all LLM strategies must implement.
It provides a common interface for working with different language models
regardless of their underlying implementation details.
"""

import os

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from src.carmina.llm.utils.prompt_loader import load_system_prompt

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider


class BaseLLMStrategy(ABC):
    """
    Abstract base class for LLM strategies.

    Defines the interface that all specific LLM implementations must follow.
    """

    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        """
        Initialize the LLM strategy.

        Args:
            model_name: Name of the specific model to use
            cloud_provider: Cloud provider instance for execution environment
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.cloud_provider = cloud_provider
        self.provider_name = self.cloud_provider.get_name()
        self.anonymization_mode = os.environ.get("ANONYMIZATION_MODE") or kwargs.get(
            "anonymization_mode", "identify"
        )
        self.temperature = float(
            os.environ.get("TEMPERATURE") or kwargs.get("temperature", 1.0)
        )
        self.max_tokens = int(
            os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens", 2500)
        )
        self.top_p = float(os.environ.get("TOP_P") or kwargs.get("top_p", 1.0))
        self.frequency_penalty = float(
            os.environ.get("FREQUENCY_PENALTY") or kwargs.get("frequency_penalty", 0.0)
        )
        self.presence_penalty = float(
            os.environ.get("PRESENCE_PENALTY") or kwargs.get("presence_penalty", 0.0)
        )

    def get_message(self, filename, text, **kwargs) -> str:
        """
        Get the message to be sent to the model.

        Args:
            filename: Name of the file being processed
            text: Text content to process
            **kwargs: Additional parameters for message formatting

        Returns:
            Formatted message string
        """
        system_prompt = load_system_prompt(f"system/{filename}")
        content = load_system_prompt(f"user/{filename}")
        text = content.replace("<input_text>", text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        return messages

    def get_inference_params(self) -> Dict[str, Any]:
        """
        Get the inference parameters for the model.

        Returns:
            Dictionary of inference parameters
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def set_anonymization_mode(self, mode: str):
        """
        Set the anonymization mode for this model.

        Args:
            mode: Anonymization mode to set (e.g., 'label', 'substitute')
        """
        self.anonymization_mode = mode

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

    @abstractmethod
    def run_inference(self, messages, inference_params) -> str:
        """
        Run inference on the model with the given messages and parameters.

        Args:
            messages: List of messages to send to the model
            inference_params: Dictionary of inference parameters

        Returns:
            Model's response as a string
        """
        pass

    @abstractmethod
    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process multiple inputs in batch mode to identify sensitive information.

        Args:
            texts: List of input texts to process
            **kwargs: Additional identification parameters

        Returns:
            List of texts with identified sensitive information
        """
        pass

    @abstractmethod
    def get_context_window(self) -> int:
        """
        Get the maximum context window size for this model.

        Returns:
            Maximum number of tokens the model can process
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: Input text to analyze

        Returns:
            Number of tokens in the text
        """
        pass

    def count_prompt_tokens(
        self, anonymization_mode: str, input_text: str
    ) -> Dict[str, int]:
        """
        Count tokens for both system and user prompts.

        Args:
            anonymization_mode: The anonymization mode (identify, label, substitute)
            input_text: The input text to be processed

        Returns:
            Dictionary with token counts for system, user, and total
        """
        try:
            # Get the messages that would be sent to the model
            messages = self.get_message(anonymization_mode, input_text)

            counts = {"system": 0, "user": 0, "total": 0}
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                tokens = self.count_tokens(content)

                if role in counts:
                    counts[role] += tokens
                counts["total"] += tokens

            return counts
        except Exception as e:
            # Fallback to simple estimation
            return {
                "system": self.count_tokens("System prompt placeholder"),
                "user": self.count_tokens(input_text),
                "total": self.count_tokens("System prompt placeholder")
                + self.count_tokens(input_text),
            }

    def get_name(self) -> str:
        """
        Get the name of this model.

        Returns:
            Model name
        """
        return self.model_name

    def get_provider(self) -> str:
        """
        Get the cloud provider name.

        Returns:
            Cloud provider name
        """
        return self.cloud_provider.get_name()

    def process_for_anonymization(self, text: str, strategy: str) -> str:
        """
        Process text specifically for anonymization tasks.

        Args:
            text: Text to anonymize
            strategy: Anonymization strategy to apply (e.g., 'label', 'substitute')

        Returns:
            Dictionary with processed results including the anonymized text
            and any identified entities
        """
        messages = self.get_message(strategy, text)
        inference_params = self.get_inference_params()
        return self.run_inference(messages, inference_params)

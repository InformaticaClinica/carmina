import logging
import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.utils.token_counter import get_token_counter


class QwenStrategy(BaseLLMStrategy):
    """
    Implementation for Qwen models.
    """

    # Dictionary to map model names to their context windows
    _context_windows = {
        "qwen2.5-72b-instruct": 131072,
        "qwen2.5-32b-instruct": 131072,
        "qwen2.5-14b-instruct": 131072,
        "qwen2.5-7b-instruct": 131072,
        "qwen2.5-3b-instruct": 32768,
        "qwen2.5-1.5b-instruct": 32768,
        "qwen2.5-0.5b-instruct": 32768,
        # Qwen 3.5
        "qwen-3.5-397b": 131072,
        "qwen-3.5-122b": 131072,
        "qwen-3.5-35b":  32768,
        "qwen-3.5-27b":  131072,
        # Qwen 3-next
        "qwen-3-next-80b": 131072,
        # Qwen 3
        "qwen-3-32b":  32768,
        "qwen-3-4b":   32768,
        "qwen-3-1.7b": 32768,
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.anonymization_mode = os.environ.get("ANONYMIZATION_MODE") or kwargs.get(
            "anonymization_mode", "label"
        )
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get(
            "temperature", 0.7
        )
        self.max_tokens = int(os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens", 2048))
        self.top_p = os.environ.get("TOP_P") or kwargs.get("top_p", 1.0)
        self.frequency_penalty = os.environ.get("FREQUENCY_PENALTY") or kwargs.get(
            "frequency_penalty", 0.0
        )
        self.presence_penalty = os.environ.get("PRESENCE_PENALTY") or kwargs.get(
            "presence_penalty", 0.0
        )
        self.model_name = model_name
        self.cloud_provider = cloud_provider
        self.provider_name = self.cloud_provider.get_name()
        self.token_counter = get_token_counter(self.model_name, "qwen")

    def run_inference(self, messages, inference_params):
        """
        Run inference on the model with the provided messages and parameters.

        Args:
            messages (list): List of messages to send to the model.
            inference_params (dict): Inference parameters for the model.

        Returns:
            str: The model's response.
        """
        inference_params = {
            k: v
            for k, v in inference_params.items()
            if v is not None and (not hasattr(v, "__len__") or len(v) > 0)
        }
        if self.provider_name == "aws":
            raise NotImplementedError(
                f"Provider {self.provider_name} not implemented for Qwen."
            )
        elif self.provider_name == "azure":
            raise NotImplementedError(
                f"Provider {self.provider_name} not implemented for Qwen."
            )
        elif self.provider_name == "google_ai_studio":
            raise NotImplementedError(
                f"Provider {self.provider_name} not implemented for Qwen."
            )
        elif self.provider_name in ("local", "ollama"):
            if "qwen" not in self.model_name.lower():
                raise ValueError(
                    f"Provider {self.provider_name} not supported for Qwen."
                )
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                },
            )
            return self.adapt_respose(response)
        else:
            raise ValueError(f"Provider {self.provider_name} not supported for Gemini.")

    def identify(self, text, **kwargs):
        message = self.get_message("identify", text)  # preguntar
        inference_params = self.get_inference_params()
        return self.run_inference(messages=message, inference_params=inference_params)

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
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
        return self._context_windows.get(self.model_name, 4096)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using Qwen tokenizer.

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

    def adapt_respose(self, response):
        idx = response.find("</think>")
        if idx != -1:
            think_start = response.find("<think>")
            think_content = response[think_start:idx + 8] if think_start != -1 else response[:idx + 8]
            logging.debug(f"[THINKING]\n{think_content}\n[/THINKING]")
            # qwen adds a few newlines so we remove them as well
            response = response[idx + 8 :].lstrip()
        return response

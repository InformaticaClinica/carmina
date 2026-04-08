import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.utils.token_counter import get_token_counter


class GLMStrategy(BaseLLMStrategy):
    """
    Implementation for GLM (General Language Model) models.
    Supports models like glm-4.7-flash running via Ollama.
    """

    # Dictionary to map model names to their context windows
    _context_windows = {
        "glm-5.1": 262144,
        "glm-5":   262144,
        "glm-4.7": 131072,
        "glm-4.6": 131072,
        "glm-4":   131072,
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.anonymization_mode = os.environ.get("ANONYMIZATION_MODE") or kwargs.get(
            "anonymization_mode", "label"
        )
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get(
            "temperature", 0.7
        )
        self.max_tokens = os.environ.get("MAX_TOKENS") or kwargs.get("max_tokens", 2500)
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
        self.token_counter = get_token_counter(self.model_name, "glm")

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
                f"Provider {self.provider_name} not implemented for GLM."
            )
        elif self.provider_name == "azure":
            raise NotImplementedError(
                f"Provider {self.provider_name} not implemented for GLM."
            )
        elif self.provider_name == "google_ai_studio":
            raise NotImplementedError(
                f"Provider {self.provider_name} not implemented for GLM."
            )
        elif self.provider_name in ("local", "ollama"):
            if "glm" not in self.model_name.lower():
                raise ValueError(
                    f"Provider {self.provider_name} not supported for GLM."
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
            # LocalProvider already extracts the message content, so just return it
            return response
        else:
            raise ValueError(f"Provider {self.provider_name} not supported for GLM.")

    def identify(self, text, **kwargs):
        message = self.get_message("identify", text)
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
        try:
            from src.carmina.llm.model_config import MODEL_CONFIGS
            model_name_lower = self.model_name.lower()
            if model_name_lower in MODEL_CONFIGS:
                return MODEL_CONFIGS[model_name_lower]["context_window"]
        except ImportError:
            pass
        return self._context_windows.get(self.model_name, 131072)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text using GLM tokenizer.

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

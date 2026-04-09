import logging
import os
from typing import List

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.token_counter import get_token_counter


class GemmaStrategy(BaseLLMStrategy):
    """
    Implementation for Gemma 4 models via local Ollama or remote Ollama API.
    """

    _context_windows = {
        # Gemma 4
        "gemma-4-31b":  262144,
        "gemma-4-26b":  262144,
        # Gemma 3
        "gemma-3-27b": 131072,
        "gemma-3-12b": 131072,
        "gemma-3-4b":  131072,
        "gemma-3-1b":  32768,
    }

    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.token_counter = get_token_counter(self.model_name, self.provider_name)

    def run_inference(self, messages, inference_params):
        inference_params = {
            k: v
            for k, v in inference_params.items()
            if v is not None and (not hasattr(v, "__len__") or len(v) > 0)
        }
        if self.provider_name in ("local", "ollama"):
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
            return self._adapt_response(response)
        raise NotImplementedError(
            f"Provider '{self.provider_name}' is not supported for Gemma models."
        )

    def identify(self, text: str, **kwargs) -> str:
        messages = self.get_message("identify", text)
        inference_params = self.get_inference_params()
        return self.run_inference(messages=messages, inference_params=inference_params)

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        pass

    def get_context_window(self) -> int:
        return self._context_windows.get(self.model_name, 131072)

    def count_tokens(self, text: str) -> int:
        return self.token_counter.count_tokens(text)

    def process_for_anonymization(self, text: str, strategy: str) -> str:
        messages = self.get_message(strategy, text)
        inference_params = self.get_inference_params()
        return self.run_inference(messages, inference_params)

    def _adapt_response(self, response: str) -> str:
        """Strip <think>…</think> blocks, logging them at DEBUG level."""
        idx = response.find("</think>")
        if idx != -1:
            think_start = response.find("<think>")
            think_content = (
                response[think_start : idx + 8]
                if think_start != -1
                else response[: idx + 8]
            )
            logging.debug(f"[THINKING]\n{think_content}\n[/THINKING]")
            response = response[idx + 8 :].lstrip()
        return response

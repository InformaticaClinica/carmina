import logging
import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.token_counter import get_token_counter


class MinimaxStrategy(BaseLLMStrategy):
    """
    Implementation for MiniMax models (e.g. minimax-m2.7) via Ollama.
    """

    _context_windows = {
        "minimax-m2.7": 1000000,
    }

    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)
        self.anonymization_mode = os.environ.get("ANONYMIZATION_MODE") or kwargs.get(
            "anonymization_mode", "label"
        )
        self.temperature = os.environ.get("TEMPERATURE") or kwargs.get("temperature", 0.7)
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
        self.token_counter = get_token_counter(self.model_name, "minimax")

    def run_inference(self, messages: List[Dict[str, str]], inference_params: dict) -> str:
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
            f"Provider '{self.provider_name}' is not supported for MiniMax models."
        )

    def _adapt_response(self, response: str) -> str:
        """Strip <think>…</think> blocks from the response, logging them at DEBUG level."""
        if not response or "<think>" not in response:
            return response
        import re
        think_blocks = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)
        for block in think_blocks:
            logging.debug(f"[THINKING]\n<think>{block}</think>\n[/THINKING]")
        return re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

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

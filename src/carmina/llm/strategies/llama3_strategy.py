import os
from typing import List, Dict, Any

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.llm.utils.prompt_loader import load_system_prompt
from src.carmina.llm.model_config import MODEL_CONFIGS

class Llama3Strategy(BaseLLMStrategy):
    """
    Implementation for Llama 3.3 models (AWS Bedrock).
    """
    _context_windows = {
        "llama-3.3-70b": 131072,  # 128K context window
    }

    def __init__(self, model_name, cloud_provider, **kwargs):
        super().__init__(model_name, cloud_provider, **kwargs)

    def identify(self, text: str, **kwargs) -> str:
        system_prompt = load_system_prompt("identify")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        if self.provider_name in ["aws", "mock", "local", "mock_ollama"]:
            if self.provider_name in ["mock", "mock_ollama"]:
                # MockProvider and MockOllamaProvider expect messages directly as a list
                response = self.cloud_provider.run_inference(
                    model_id=self.model_name,
                    messages=messages,
                    inference_params={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty
                    }
                )
            elif self.provider_name == "local":
                # LocalProvider expects messages directly as a list
                response = self.cloud_provider.run_inference(
                    model_id=self.model_name,
                    messages=messages,
                    inference_params={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty
                    }
                )
            else:
                # AWS provider expects messages wrapped in a dict
                response = self.cloud_provider.run_inference(
                    model_id=self.model_name,
                    messages={"messages": messages},
                    inference_params={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "frequency_penalty": self.frequency_penalty,
                        "presence_penalty": self.presence_penalty
                    }
                )
            # AWSProvider returns a dict, extract the content properly
            if isinstance(response, dict):
                if "generation" in response:
                    return response["generation"]
                elif "content" in response:
                    return response["content"]
                elif "output" in response:
                    return response["output"]
            return response
        else:
            raise ValueError(f"Unsupported {self.provider_name} provider type for identification")

    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        results = []
        for text in texts:
            result = self.identify(text, **kwargs)
            results.append(result)
        return results

    def get_context_window(self) -> int:
        return self._context_windows.get(self.model_name, 8192)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def run_inference(self, messages, inference_params) -> str:
        if self.provider_name in ["aws", "mock", "local", "mock_ollama"]:
            response = self.cloud_provider.run_inference(
                model_id=self.model_name,
                messages=messages,
                inference_params=inference_params
            )
            # Handle response format for different providers
            if isinstance(response, dict):
                if "generation" in response:
                    return response["generation"]
                elif "content" in response:
                    return response["content"]
                elif "output" in response:
                    return response["output"]
            return response
        else:
            raise ValueError(f"Unsupported {self.provider_name} provider type for Llama3Strategy")

    def process_for_anonymization(self, text: str, strategy: str) -> Dict[str, Any]:
        messages = self.get_message(strategy, text)
        inference_params = self.get_inference_params()
        result = self.run_inference(messages, inference_params)
        return result

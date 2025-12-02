import os
import logging
import requests
from typing import List, Dict
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider


class LocalProvider(BaseCloudProvider):
    """
    Local provider implementation for running inference on local models with Ollama.

    This provider connects to a local Ollama instance to run models like Llama,
    Gemma, etc. directly on the user's machine.
    """

    # Mapping for local models to their Ollama tags
    _local_model_ids = {
        # Llama models
        "llama-3.1-8b": "llama3.1:8b",
        "llama-3.1-70b": "llama3.1:70b",
        "llama-3.1-405b": "llama3.1:405b",
        "llama-3.2-1b": "llama3.2:1b",
        "llama-3.2-3b": "llama3.2:3b",
        "llama-3.3-8b": "llama3.3:8b",
        "llama-3.3-70b": "llama3.3:70b",
        "llama-4": "llama4",
        "llama-4-scout": "llama4:scout",
        # Gemma models
        "gemma-3-1b": "gemma3:1b",
        "gemma-3-4b": "gemma3:4b",
        "gemma-3-27b": "gemma3:27b",
        # Qwen models
        "qwen-3-1.7b": "qwen3:1.7b",
        "qwen-3-32b": "qwen3:32b",
        # Mistral models
        "mistral": "mistral:latest",
        # Deepseek models
        "deepseek-r1-1.5b": "deepseek-r1:1.5b",
        # GPT-OSS models
        "gpt-oss-120b": "gpt-oss:120b",
    }

    def __init__(self, name="local", **kwargs):
        """
        Initialize the LocalProvider with Ollama endpoint.
        """
        super().__init__(name=name, **kwargs)
        self.base_url = os.environ.get("OLLAMA_BASE_URL") or kwargs.get("base_url")
        self.api_endpoint = f"{self.base_url}/api/chat"
        logging.info(f"LocalProvider initialized with endpoint {self.base_url}")

    def connect(self):
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.base_url}/api/version")
            if response.status_code == 200:
                version = response.json().get("version", "unknown")
                logging.info(f"Connected to Ollama version {version}")
                return f"Connected to Ollama version"
            else:
                logging.warning(f"Could not connect to Ollama: {response.status_code}")
        except Exception as e:
            logging.warning(f"Error connecting to Ollama: {e}")

    def run_inference(self, model_id: str, messages: List[Dict[str, str]], **kwargs):
        """
        Run inference on a local model via Ollama.

        Args:
            model_id (str): The model ID to use (e.g., "llama-3.2-3b").
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional parameters like temperature, max_tokens, etc.

        Returns:
            str: The generated text response.
        """
        logging.info(f"Running inference on local model {model_id}")

        # Get the Ollama model name
        ollama_model = self.get_model_id(model_id)

        # Extract inference parameters
        inference_params = kwargs.get("inference_params", {})

        payload = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "temperature": inference_params.get("temperature"),
            "num_predict": inference_params.get("max_tokens"),
            "top_p": inference_params.get("top_p"),
            "top_k": 0,
        }

        try:
            logging.debug(
                f"Sending request to {self.api_endpoint} with payload: {payload}"
            )
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()

            result = response.json()
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            else:
                logging.error(f"Unexpected response structure: {result}")
                return ""

        except requests.exceptions.RequestException as e:
            logging.error(f"Error during Ollama inference: {e}")
            if hasattr(e, "response") and e.response:
                logging.error(f"Response: {e.response.text}")
            return f"Error: Could not complete inference with local model. {str(e)}"

    def get_model_id(self, model_name: str) -> str:
        """
        Returns the Ollama model name for the given model ID.
        """
        return self._local_model_ids.get(model_name, model_name)

    def get_name(self) -> str:
        """
        Returns the name of the cloud provider.
        """
        return "local"

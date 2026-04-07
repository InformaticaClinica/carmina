import os
import logging
import time
import requests
from typing import List, Dict
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider
from src.carmina.llm.utils.ollama_watchdog import (
    is_bad_response,
    restart_ollama,
    unload_model,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RESTART_WAIT,
    DEFAULT_MAX_RETRIES,
)


class LocalProvider(BaseCloudProvider):
    """
    Local provider implementation for running inference on local models with Ollama.

    This provider connects to a local Ollama instance to run models like Llama,
    Gemma, etc. directly on the user's machine.

    Resilience features
    -------------------
    * Per-request HTTP timeout (OLLAMA_REQUEST_TIMEOUT env var, default 120 s).
    * Automatic retry on timeout/connection-error with full Ollama restart and
      VRAM eviction between attempts.
    * Bad-response detection: responses that echo the system prompt or contain
      unresolved template tokens are treated as failures and retried.
    * Configurable max retries (OLLAMA_MAX_RETRIES env var, default 3).
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
        "qwen-3-4b": "qwen3:4b",
        # Qwen 3.5 models
        "qwen-3.5-27b": "qwen3.5:27b",
        "qwen-3.5-35b": "qwen3.5:35b",
        "qwen-3.5-122b": "qwen3.5:122b",
        # GLM models
        "glm-4.7-flash": "glm-4.7-flash:q4_K_M",
        "glm4.7-flash": "glm-4.7-flash:q4_K_M",
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

        # Resilience tunables
        self.request_timeout = int(
            os.environ.get("OLLAMA_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
        )
        self.restart_wait = int(
            os.environ.get("OLLAMA_RESTART_WAIT", DEFAULT_RESTART_WAIT)
        )
        self.max_retries = int(
            os.environ.get("OLLAMA_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )

        logging.info(
            f"LocalProvider initialised — endpoint={self.base_url}  "
            f"timeout={self.request_timeout}s  max_retries={self.max_retries}"
        )

    def connect(self):
        """Test connection to Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=10)
            if response.status_code == 200:
                version = response.json().get("version", "unknown")
                logging.info(f"Connected to Ollama version {version}")
                return "Connected to Ollama version"
            else:
                logging.warning(f"Could not connect to Ollama: {response.status_code}")
        except Exception as e:
            logging.warning(f"Error connecting to Ollama: {e}")

    # ------------------------------------------------------------------
    # Public inference entry-point with retry/restart loop
    # ------------------------------------------------------------------

    def run_inference(self, model_id: str, messages: List[Dict[str, str]], **kwargs):
        """
        Run inference on a local model via Ollama with automatic stall recovery.

        On timeout or connection error the method:
          1. Unloads the model from VRAM (clean eviction).
          2. Restarts the Ollama process (full VRAM/RAM flush).
          3. Retries the request up to *max_retries* times.

        Responses that contain echoed prompt fragments or unresolved template
        tokens are also considered failures and trigger the same recovery flow.

        Args:
            model_id:  Logical model ID (e.g. "qwen-3.5-122b").
            messages:  Chat messages list with 'role'/'content' dicts.
            **kwargs:  Forwarded to payload; supports 'inference_params'.

        Returns:
            str: Model response content, or "" on unrecoverable failure.
        """
        logging.info(f"Running inference on local model {model_id}")

        ollama_model = self.get_model_id(model_id)
        inference_params = kwargs.get("inference_params", {})
        payload = self._build_payload(ollama_model, messages, inference_params)

        last_error: str = ""
        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                logging.warning(
                    f"LocalProvider: attempt {attempt}/{self.max_retries} "
                    f"for model {ollama_model} — reason: {last_error}"
                )
                # Free VRAM first (clean unload), then hard-restart
                unload_model(self.base_url, ollama_model)
                restarted = restart_ollama(self.base_url, wait=self.restart_wait)
                if not restarted:
                    logging.error(
                        "LocalProvider: Ollama did not come back after restart — "
                        "sleeping 30 s before next attempt."
                    )
                    time.sleep(30)

            result = self._do_request(payload)

            if result is None:
                # Hard failure (timeout / connection error) — retry
                last_error = "request timeout or connection error"
                continue

            if result == "":
                # Empty response is valid (no entities) — return immediately
                return ""

            bad, reason = is_bad_response(result)
            if bad:
                last_error = reason
                logging.warning(
                    f"LocalProvider: bad response detected ({reason}). "
                    f"Will retry (attempt {attempt}/{self.max_retries})."
                )
                continue

            # All good
            return result

        logging.error(
            f"LocalProvider: all {self.max_retries} attempts failed for model "
            f"{ollama_model}. Last error: {last_error}. Returning empty string."
        )
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        ollama_model: str,
        messages: List[Dict[str, str]],
        inference_params: dict,
    ) -> dict:
        payload = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
            "temperature": inference_params.get("temperature"),
            "num_predict": inference_params.get("max_tokens"),
            "top_p": inference_params.get("top_p"),
            "top_k": 0,
        }
        if "think" in inference_params:
            payload["think"] = inference_params["think"]
        return payload

    def _do_request(self, payload: dict):
        """
        Execute a single POST to /api/chat.

        Returns:
            str   — response content on success.
            ""    — on an unexpected but non-fatal response structure.
            None  — on timeout or connection error (signals a retry).
        """
        try:
            logging.debug(
                f"sending request to {self.api_endpoint} with payload: {payload}"
            )
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            result = response.json()
            if "message" in result and "content" in result["message"]:
                message = result["message"]
                content = message["content"]
                thinking = message.get("thinking", "")
                # Re-wrap thinking so downstream adapt_response can handle it
                if thinking:
                    content = f"<think>{thinking}</think>\n{content}"
                return content

            logging.error(f"LocalProvider: unexpected response structure: {result}")
            return ""

        except requests.exceptions.Timeout:
            logging.error(
                f"LocalProvider: request timed out after {self.request_timeout}s "
                f"(model={payload.get('model')}) — scheduling restart."
            )
            return None

        except requests.exceptions.ConnectionError as e:
            logging.error(f"LocalProvider: connection error — {e}")
            return None

        except requests.exceptions.RequestException as e:
            logging.error(f"LocalProvider: HTTP error — {e}")
            if hasattr(e, "response") and e.response is not None:
                logging.error(f"Response body: {e.response.text[:500]}")
            return None

    def get_model_id(self, model_name: str) -> str:
        """
        Returns the Ollama model name for the given model ID.
        """
        return self._local_model_ids.get(model_name, model_name)

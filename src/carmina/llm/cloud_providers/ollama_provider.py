"""
Ollama Cloud Provider for LLM operations.

Connects to a remote Ollama API server with API-key authentication.
Unlike LocalProvider (which manages a local Ollama process), this provider
targets a hosted/remote Ollama endpoint and therefore has no watchdog or
local-process-restart logic.
"""

import os
import logging
import requests
from typing import List, Dict, Optional

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider


class OllamaProvider(BaseCloudProvider):
    """
    Provider for remote Ollama API instances with API-key authentication.

    Suitable for hosted Ollama deployments (e.g. ollama.com or a self-hosted
    instance protected by a reverse proxy).

    Model mapping
    -------------
    Use the canonical model name (e.g. "qwen-3.5-122b" or "qwen-3.5-122b-think")
    in the MODELS env var.  The ``-think`` suffix signals that the thinking/
    chain-of-thought mode should be enabled for models that support it
    (Qwen 3.5 and Gemma 4).
    """

    # Canonical model name → Ollama model tag
    # Tags verified against api.ollama.com /api/tags (2026-04-08)
    _model_ids: Dict[str, str] = {
        # ── Qwen 3.5 ──────────────────────────────────────────────────────────
        "qwen-3.5-397b": "qwen3.5:397b",
        "qwen-3.5-122b": "qwen3.5:122b",
        "qwen-3.5-35b":  "qwen3.5:35b",
        "qwen-3.5-27b":  "qwen3.5:27b",
        # ── Qwen 3-next ───────────────────────────────────────────────────────
        "qwen-3-next-80b": "qwen3-next:80b",
        # ── Qwen 3 ────────────────────────────────────────────────────────────
        "qwen-3-32b":  "qwen3:32b",
        "qwen-3-4b":   "qwen3:4b",
        "qwen-3-1.7b": "qwen3:1.7b",
        # ── Gemma 4 ───────────────────────────────────────────────────────────
        "gemma-4-31b":  "gemma4:31b",
        "gemma-4-26b":  "gemma4:26b",
        # ── Gemma 3 ───────────────────────────────────────────────────────────
        "gemma-3-27b": "gemma3:27b",
        "gemma-3-12b": "gemma3:12b",
        "gemma-3-4b":  "gemma3:4b",
        "gemma-3-1b":  "gemma3:1b",
        # ── GLM ───────────────────────────────────────────────────────────────
        "glm-5.1": "glm-5.1",
        "glm-5":   "glm-5",
        "glm-4.7": "glm-4.7",
        "glm-4.6": "glm-4.6",
        "glm-4":   "glm-4",
        # ── MiniMax ───────────────────────────────────────────────────────────
        "minimax-m2.7": "minimax-m2.7",
        "minimax-m2.5": "minimax-m2.5",
        "minimax-m2.1": "minimax-m2.1",
        "minimax-m2":   "minimax-m2",
        # ── GPT-OSS ───────────────────────────────────────────────────────────
        "gpt-oss-20b":  "gpt-oss:20b",
        "gpt-oss-120b": "gpt-oss:120b",
        # ── DeepSeek ──────────────────────────────────────────────────────────
        "deepseek-v3.2":      "deepseek-v3.2",
        "deepseek-v3.1-671b": "deepseek-v3.1:671b",
        # ── Kimi ──────────────────────────────────────────────────────────────
        "kimi-k2-1t":      "kimi-k2:1t",
        "kimi-k2.5":       "kimi-k2.5",
        "kimi-k2-thinking": "kimi-k2-thinking",
        # ── Llama ─────────────────────────────────────────────────────────────
        "llama-3.1-8b":  "llama3.1:8b",
        "llama-3.1-70b": "llama3.1:70b",
        "llama-3.3-70b": "llama3.3:70b",
        "llama-4":       "llama4",
        "llama-4-scout": "llama4:scout",
    }

    DEFAULT_REQUEST_TIMEOUT = 300

    def __init__(self, name: str = "ollama", **kwargs):
        """
        Initialise the OllamaProvider.

        Reads from environment variables (highest priority) then kwargs:
          OLLAMA_API_URL          — base URL of the Ollama API server
          OLLAMA_API_KEY          — Bearer token for authentication
          OLLAMA_REQUEST_TIMEOUT  — per-request HTTP timeout in seconds
          OLLAMA_MAX_RETRIES      — number of retry attempts on transient errors
        """
        super().__init__(name=name, **kwargs)
        self.base_url: str = (
            os.environ.get("OLLAMA_API_URL")
            or kwargs.get("base_url", "https://api.ollama.com")
        )
        self.api_key: Optional[str] = (
            os.environ.get("OLLAMA_API_KEY") or kwargs.get("api_key")
        )
        self.api_endpoint: str = f"{self.base_url}/api/chat"
        self.request_timeout: int = int(
            os.environ.get("OLLAMA_REQUEST_TIMEOUT", self.DEFAULT_REQUEST_TIMEOUT)
        )
        self.max_retries: int = int(os.environ.get("OLLAMA_MAX_RETRIES", 3))

        self._headers: Dict[str, str] = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

        logging.info(
            f"OllamaProvider initialised — endpoint={self.base_url} "
            f"timeout={self.request_timeout}s  max_retries={self.max_retries}"
        )

    # ------------------------------------------------------------------
    # BaseCloudProvider interface
    # ------------------------------------------------------------------

    def connect(self) -> str:
        """Test connectivity to the Ollama API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                headers=self._headers,
                timeout=10,
            )
            response.raise_for_status()
            version = response.json().get("version", "unknown")
            logging.info(f"OllamaProvider: connected — version={version}")
            return f"Connected to Ollama API version {version}"
        except Exception as exc:
            logging.warning(f"OllamaProvider: could not connect — {exc}")
            return f"Connection failed: {exc}"

    def run_inference(
        self, model_id: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """
        Run inference via the remote Ollama /api/chat endpoint.

        Args:
            model_id:         Logical model name (e.g. "qwen-3.5-122b-think").
            messages:         Chat messages list with 'role'/'content' dicts.
            **kwargs:         Accepts 'inference_params' dict with keys:
                              temperature, max_tokens, top_p, think, etc.

        Returns:
            Model response as a string (may be "" for empty valid responses).
        """
        logging.info(f"OllamaProvider: inference for model={model_id}")

        ollama_model = self.get_model_id(model_id)
        inference_params = kwargs.get("inference_params", {})
        payload = self._build_payload(ollama_model, messages, inference_params)

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                logging.warning(
                    f"OllamaProvider: retry {attempt}/{self.max_retries} "
                    f"for {ollama_model} — {last_error}"
                )

            result = self._do_request(payload)

            if result is None:
                last_error = "request timeout or connection error"
                continue

            return result  # "" (valid empty) or actual content

        logging.error(
            f"OllamaProvider: all {self.max_retries} attempts failed "
            f"for {ollama_model}. Last error: {last_error}"
        )
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        ollama_model: str,
        messages: List[Dict],
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

    def _do_request(self, payload: dict) -> Optional[str]:
        """
        Execute a single POST to /api/chat.

        Returns:
            str  — response content on success (may be "").
            None — on timeout or connection error (triggers a retry).
        """
        try:
            logging.debug(
                f"OllamaProvider: POST {self.api_endpoint} "
                f"model={payload.get('model')}"
            )
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=self._headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            result = response.json()
            if "message" in result and "content" in result["message"]:
                message = result["message"]
                content = message["content"]
                thinking = message.get("thinking", "")
                if thinking:
                    content = f"<think>{thinking}</think>\n{content}"
                return content

            logging.error(
                f"OllamaProvider: unexpected response structure: {result}"
            )
            return ""

        except requests.exceptions.Timeout:
            logging.error(
                f"OllamaProvider: request timed out after {self.request_timeout}s "
                f"(model={payload.get('model')})"
            )
            return None

        except requests.exceptions.ConnectionError as exc:
            logging.error(f"OllamaProvider: connection error — {exc}")
            return None

        except requests.exceptions.RequestException as exc:
            logging.error(f"OllamaProvider: HTTP error — {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                logging.error(f"Response body: {exc.response.text[:500]}")
            return None

    def get_model_id(self, model_name: str) -> str:
        """Return the Ollama tag for a logical model name."""
        return self._model_ids.get(model_name, model_name)

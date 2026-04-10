import os
import json
import logging
import threading
from pathlib import Path

from google import genai
from google.genai import types
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import ResourceExhausted, InternalServerError, ServiceUnavailable
from google.genai.errors import ClientError as GenAIClientError, ServerError as GenAIServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
from typing import Optional, Dict, Any, List
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider


class SafetyBlockError(Exception):
    """Raised when Vertex AI blocks a response due to safety filters."""
    pass


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient errors that should be retried.

    Covers both google-api-core and google-genai exception hierarchies,
    since the latter raises its own ClientError/ServerError types that do
    NOT inherit from google.api_core.exceptions.
    """
    # google-api-core exceptions (legacy / some SDK paths)
    if isinstance(exc, (ResourceExhausted, InternalServerError, ServiceUnavailable)):
        return True
    # google-genai exceptions (google.genai.Client path)
    if isinstance(exc, GenAIClientError) and exc.code == 429:
        return True
    if isinstance(exc, GenAIServerError):
        return True
    return False


class VertexAIProvider(BaseCloudProvider):
    """
    Vertex AI provider for running inference on models.
    
    This class is designed to interact with Google Cloud Vertex AI's API for running inference
    on various models. It provides methods to initialize the provider, run inference, and
    manage model IDs.
    
    Features:
    - Automatic retry with exponential backoff for transient errors (429, 5xx)
    - External model ID mapping configuration
    - Fallback strategy for unmapped models
    - Safety block detection and explicit error handling
    """
    
    # Model ID mappings (loaded from config or fallback to defaults)
    _vertex_model_ids = None
    _config_loaded = False
    _config_lock = threading.Lock()  # Guards class-level config initialisation
    
    def __init__(self, **kwargs):
        self.project_id = kwargs.get("project_id") or os.environ.get("VERTEX_PROJECT_ID")
        self.location = kwargs.get("location") or os.environ.get("VERTEX_LOCATION", "global")
        self.timeout = kwargs.get("timeout", 60)  # Default 60 seconds timeout
        
        if not self.project_id:
            raise ValueError("Project ID is required for Vertex AI provider.")
        
        # Set GOOGLE_APPLICATION_CREDENTIALS from .env if not already set
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and not os.path.isabs(creds_path):
            # If it's a relative path, make it absolute from project root
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            abs_creds_path = project_root / creds_path
            if abs_creds_path.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(abs_creds_path)
                logging.info(f"Using credentials from: {abs_creds_path}")
            else:
                logging.warning(f"Credentials file not found at: {abs_creds_path}")
        
        try:
            credentials, project = default()
        except DefaultCredentialsError as e:
            raise ValueError("Google Cloud credentials not found.") from e
        
        self._client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        
        # Load model mappings from config on first initialization (thread-safe)
        with VertexAIProvider._config_lock:
            if not VertexAIProvider._config_loaded:
                self._load_model_mappings()
        
        logging.info(f"Vertex AI Provider initialized (project={self.project_id}, location={self.location})")
    
    @classmethod
    def _load_model_mappings(cls):
        """Load model ID mappings from external config file with fallback to defaults."""
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "vertex_model_mappings.json"
        
        default_mappings = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-3.0-pro":"gemini-3-pro-preview",
            "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
            "gemini-3-flash" : "gemini-3-flash-preview"
        }
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    cls._vertex_model_ids = config.get("mappings", default_mappings)
                    logging.info(f"Loaded model mappings from {config_path}")
            else:
                cls._vertex_model_ids = default_mappings
                logging.warning(f"Model mappings config not found at {config_path}, using defaults")
        except Exception as e:
            logging.error(f"Failed to load model mappings config: {e}, using defaults")
            cls._vertex_model_ids = default_mappings
        
        cls._config_loaded = True
    
    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True
    )
    def run_inference(self, model_id: str, messages: dict, **kwargs) -> str:
        """
        Runs inference using a model in Vertex AI with automatic retry for transient errors.
        
        Args:
            model_id (str): The ID of the model to use for inference.
            messages (dict): The input messages for the model.
            **kwargs: Additional parameters for the inference request.
        
        Returns:
            str: The response from the model.
            
        Raises:
            ValueError: If model_id is not supported and fallback fails.
            SafetyBlockError: If response is blocked by safety filters.
            ResourceExhausted: If quota is exceeded (after retries).
            InternalServerError: If server error persists (after retries).
        """
        logging.info(f"Running inference on Vertex AI with model {model_id}")
        model_name = self._vertex_model_ids.get(model_id)
        
        if not model_name:
            # Fallback strategy: use model_id as-is
            logging.warning(f"Model ID {model_id} not in mapping, using ID as-is (fallback)")
            model_name = model_id
        
        # Adapt the messages to the required format
        payload = self.adapt_message(messages, **kwargs)

        # Use streaming to avoid read timeouts on long responses
        full_text = ""
        last_candidates = None
        for chunk in self._client.models.generate_content_stream(
            model=model_name,
            contents=payload["contents"],
            config=payload["config"],
        ):
            if hasattr(chunk, 'text') and chunk.text:
                full_text += chunk.text
            if hasattr(chunk, 'candidates') and chunk.candidates:
                last_candidates = chunk.candidates

        # Check for safety blocks on the final chunk
        if last_candidates:
            candidate = last_candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason)
                if 'SAFETY' in finish_reason:
                    logging.warning(f"Response blocked by safety filters: {finish_reason}")
                    raise SafetyBlockError(f"Response blocked by safety filters: {finish_reason}")

        if not full_text:
            raise ValueError("Response does not contain text content")

        return full_text
    
    def get_name(self) -> str:
        """
        Returns the name of the cloud provider.
        
        Returns:
            str: Cloud provider name.
        """
        return "vertex_ai"
    
    def adapt_message(self, messages: list, **kwargs) -> list:
        """
        Adapts the message format for Vertex AI.
        
        Args:
            messages (list): List of messages to adapt.
        
        Returns:
            list: List of adapted messages.
        """
        payload = {}
        # Extract system instruction and user messages
        system_instruction, payload["contents"] = self.get_messages(messages)
        # Generate the payload for the inference request
        payload["config"] = self.generate_config(system_instruction, **kwargs)
        return payload
    
    def get_messages(self, messages: list) -> list:
        """
        Returns the messages in the required format for Vertex AI.
        
        Args:
            messages (list): List of messages to format.
        
        Returns:
            list: List of formatted messages.
        """
        formatted_contents = []
        system_instruction = None
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                formatted_contents.append(content)
            if role == "assistant":
                raise ValueError("Assistant role is not supported in Vertex AI.")
        contents = self.check_user_message(formatted_contents)
        return system_instruction, contents
    
    def generate_config(self, system_instruction: Optional[str], **kwargs) -> dict:
        """
        Generates the payload for the inference request.
        
        Args:
            system_instruction (Optional[str]): System instruction for the model, can be string or None.
            **kwargs: Additional configuration parameters (temperature, max_tokens, top_p, top_k, stop_sequences).
        
        Returns:
            types.GenerateContentConfig: The generated configuration.
        """
        config_params = {
            "max_output_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "stop_sequences": kwargs.get("stop_sequences")
        }
        
        # Add system_instruction if provided
        if system_instruction:
            config_params["system_instruction"] = system_instruction
        
        return types.GenerateContentConfig(**config_params)
    
    def check_user_message(self, messages: list) -> bool:
        if len(messages) > 1:
            raise ValueError("Multiple user messages are not supported in Vertex AI.")
        elif len(messages) == 0:
            raise ValueError("No user message provided.")
        elif len(messages) == 1:
            return messages[0]

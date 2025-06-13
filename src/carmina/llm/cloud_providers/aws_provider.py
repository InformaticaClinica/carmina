"""
AWS Cloud Provider for LLM operations.

This module implements the AWS-specific provider for accessing LLM models
through AWS services like Amazon Bedrock.
"""

import os
import json
import boto3
import time
import random
from typing import Dict, Any, Optional, List
import logging
from botocore.exceptions import ClientError, ReadTimeoutError
from botocore.config import Config
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider

class AWSProvider(BaseCloudProvider):
    """
    AWS provider for accessing LLMs through Amazon's cloud services.
    
    This provider enables access to models like Claude, Llama, and Mistral
    through Amazon Bedrock.
    """
    
    # Model ID mapping for AWS Bedrock models
    _bedrock_model_ids = {
        # Anthropic Claude models
        "claude-3.5-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3.7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0", #no funciona
        "claude-3.5-sonnet":"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3.5-sonnet-v2":"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-4-opus": "us.anthropic.claude-opus-4-20250514-v1:0", #no funciona
        "claude-3-haiku":"us.anthropic.claude-3-haiku-20240307-v1:0",
        "claude-3-sonnet":"us.anthropic.claude-3-sonnet-20240229-v1:0",

        
        # LLama models
        "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
        "llama-3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
        "llama-3.2-1b": "us.meta.llama3-2-1b-instruct-v1:0",
        "llama-3.2-3b": "us.meta.llama3-2-3b-instruct-v1:0", 
        # Other models
        "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
        "mistral-large":"mistral.mistral-large-2402-v1:0",
    }
    
    def __init__(self, name:str = "aws", **kwargs):
        """
        Initialize the AWS provider with appropriate credentials and configuration.
        
        Args:
            **kwargs: Additional AWS-specific configuration parameters
        """
        super().__init__(name=name, **kwargs)
        self.region = os.environ.get("AWS_REGION") or kwargs.get("region", "us-east-1")
        self.access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or kwargs.get("access_key_id", None)
        self.secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or kwargs.get("secret_access_key", None)
        self.session_token = os.environ.get("AWS_SESSION_TOKEN") or kwargs.get("session_token", None)
        self.profile = os.environ.get("AWS_PROFILE") or kwargs.get("profile", None)
        
        # Rate limiting and retry configuration
        self.max_retries = int(os.environ.get("AWS_MAX_RETRIES", "5"))
        self.base_delay = float(os.environ.get("AWS_BASE_DELAY", "2.0"))  # Increased from 1.0
        self.max_delay = float(os.environ.get("AWS_MAX_DELAY", "120.0"))  # Increased from 60.0
        self.request_delay = float(os.environ.get("AWS_REQUEST_DELAY", "1.0"))  # Increased from 0.1
        
        # Timeout configuration
        self.read_timeout = int(os.environ.get("AWS_READ_TIMEOUT", "120"))  # 2 minutes
        self.connect_timeout = int(os.environ.get("AWS_CONNECT_TIMEOUT", "60"))  # 1 minute
        
        # Initialize clients with custom config
        self.session = self._create_session()
        self.bedrock_runtime = self._create_bedrock_client()
        
        # Service flags
        self._is_initialized = False
    
    def _create_bedrock_client(self):
        """
        Create Bedrock client with custom timeout and retry configuration.
        
        Returns:
            Configured Bedrock runtime client
        """
        config = Config(
            region_name=self.region,
            retries={
                'max_attempts': 1,  # We handle retries manually
                'mode': 'standard'
            },
            read_timeout=self.read_timeout,
            connect_timeout=self.connect_timeout,
            max_pool_connections=50
        )
        
        return self.session.client(
            service_name="bedrock-runtime",
            config=config
        )
    
    def connect(self) -> str:
        """
        Establish a connection to AWS services.
        
        Returns:
            Connection status message
        """
        try:
            # Create a bedrock client for testing connectivity
            bedrock_client = self.session.client(
                service_name="bedrock",
                region_name=self.region,
                config=Config(
                    read_timeout=30,
                    connect_timeout=10
                )
            )
            bedrock_client.list_foundation_models()
            self._is_initialized = True
            return "Connected to AWS API"
        except (ClientError, ReadTimeoutError, Urllib3ReadTimeoutError) as e:
            logging.error(f"Connection error: {str(e)}")
            raise

    def _create_session(self) -> boto3.Session:
        """
        Create a properly configured boto3 session.
        
        Returns:
            Configured boto3 Session object
        """
        if self.profile:
            return boto3.Session(profile_name=self.profile, region_name=self.region)
        return boto3.Session(region_name=self.region)
        
    def get_name(self) -> str:
        """
        Get the provider name.
        
        Returns:
            String identifier for this provider ("aws")
        """
        return "aws"
    
    def get_bedrock_model_id(self, model_name: str) -> str:
        """
        Get the AWS Bedrock model ID for a given model name.
        
        Args:
            model_name: The user-friendly model name
            
        Returns:
            AWS Bedrock model ID
            
        Raises:
            ValueError: If the model is not supported in Bedrock
        """
        model_name_lower = model_name.lower()
        
        # Check for exact match
        if model_name_lower in self._bedrock_model_ids:
            return self._bedrock_model_ids[model_name_lower]
        
        # Check for partial matches
        for name_pattern, model_id in self._bedrock_model_ids.items():
            if name_pattern in model_name_lower:
                return model_id
                
        raise ValueError(
            f"Model '{model_name}' not supported in AWS Bedrock. "
            f"Supported models: {', '.join(self._bedrock_model_ids.keys())}"
        )
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic for ThrottlingException and timeouts.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add a delay before each attempt (including the first one for rate limiting)
                if attempt > 0:
                    # Exponential backoff for retries
                    delay = min(
                        self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1),
                        self.max_delay
                    )
                    logging.warning(f"Waiting {delay:.2f} seconds before retry attempt {attempt + 1}")
                    time.sleep(delay)
                else:
                    # Small delay even on first attempt to avoid rate limiting
                    time.sleep(self.request_delay)
                
                return func(*args, **kwargs)
                
            except (ClientError, ReadTimeoutError, Urllib3ReadTimeoutError) as e:
                last_exception = e
                
                # Handle different types of errors
                if isinstance(e, ClientError):
                    error_code = e.response.get('Error', {}).get('Code', '')
                    if error_code == 'ThrottlingException' and attempt < self.max_retries:
                        logging.warning(
                            f"ThrottlingException on attempt {attempt + 1}/{self.max_retries + 1}. "
                            f"Will retry..."
                        )
                        continue
                    else:
                        # Re-raise non-throttling ClientErrors or if max retries exceeded
                        if attempt == self.max_retries:
                            logging.error(f"Max retries exceeded for ClientError: {error_code}")
                        raise
                        
                elif isinstance(e, (ReadTimeoutError, Urllib3ReadTimeoutError)) and attempt < self.max_retries:
                    logging.warning(
                        f"Read timeout on attempt {attempt + 1}/{self.max_retries + 1}. "
                        f"Will retry with longer timeout..."
                    )
                    # Recreate client with longer timeout for next attempt
                    self.read_timeout = min(self.read_timeout * 1.5, 300)  # Max 5 minutes
                    self.bedrock_runtime = self._create_bedrock_client()
                    continue
                else:
                    # Re-raise timeout errors if max retries exceeded
                    if attempt == self.max_retries:
                        logging.error("Max retries exceeded for timeout error")
                    raise
                    
            except Exception as e:
                # Re-raise other exceptions immediately
                logging.error(f"Unexpected error: {str(e)}")
                raise
        
        # If we get here, all retries were exhausted
        raise last_exception
    
    def run_inference(
        self,
        model_id: str,
        messages: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute inference using AWS Bedrock.

        Args:
            model_id: Name of the model to use
            messages: Input data for the model
            **kwargs: Additional parameters for inference
            
        Returns:
            Model response as a dictionary
            
        Raises:
            ValueError: If the model or request is invalid
            ClientError: If AWS returns an error
        """
        inference_params = kwargs.get("inference_params", {})
        
        try:
            model_id = self.get_bedrock_model_id(model_id)
            
            # Format the request according to the provider's requirements
            if "anthropic" in model_id:
                messages = messages.get("messages", [])
                
                # Format for Claude models
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": int(inference_params.get("max_tokens", 4096)),
                    "temperature": float(inference_params.get("temperature", 0.1)),
                }
                
                # Separate system and user messages
                system_messages = [msg for msg in messages if msg.get("role") == "system"]
                user_messages = [msg for msg in messages if msg.get("role") != "system"]
                
                if system_messages:
                    request_body["system"] = system_messages[0].get("content", "")
                
                request_body["messages"] = user_messages
                    
            elif "meta.llama" in model_id:
                messages_list = messages
                # Convert messages to prompt format for Llama
                prompt_parts = []
                for msg in messages_list:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
                    elif role == "user":
                        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
                    elif role == "assistant":
                        prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")
                
                prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
                prompt = "".join(prompt_parts)
                
                request_body = {
                    "prompt": prompt,
                    "max_gen_len": inference_params.get("max_tokens", 2000),
                    "temperature": inference_params.get("temperature", 0.6),
                    "top_p": inference_params.get("top_p", 0.9),
                }
            elif "mistral-large" in model_id:
                # New Mistral Large format (role-based)
                request_body = {
                    "messages": messages,
                    "temperature": inference_params.get("temperature", 0.6),
                    "top_p": inference_params.get("top_p", 0.9),
                    "max_tokens": inference_params.get("max_tokens", 2048),
                }
            elif "mistral" in model_id:
                # Mistral 7B expects a simple prompt format
                messages_list = messages if isinstance(messages, list) else []
                system = next((m["content"] for m in messages_list if m.get("role") == "system"), "")
                user = next((m["content"] for m in messages_list if m.get("role") == "user"), "")
                
                prompt = f"[INST] {system}\n{user} [/INST]"
                request_body = {
                    "prompt": prompt,
                    "max_tokens": inference_params.get("max_tokens", 2048),
                    "temperature": inference_params.get("temperature", 0.6)
                }   
            logging.debug(f"Request body for {model_id}: {json.dumps(request_body, indent=2)}")
                    
            # Make the actual API call with retry logic
            def _invoke_model():
                return self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
            
            response = self._retry_with_backoff(_invoke_model)
            response_body = json.loads(response.get('body').read())
            return response_body
            
        except (ClientError, ReadTimeoutError, Urllib3ReadTimeoutError) as e:
            if isinstance(e, ClientError):
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ValidationException':
                    logging.error(f"Request validation error: {str(e)}")
                    logging.error(f"Request body was: {json.dumps(request_body, indent=2)}")
            else:
                logging.error(f"Timeout error: {str(e)}")
            raise
    
    def batch_process(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, Any]],
        inference_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch mode with aggressive rate limiting.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input data dictionaries
            inference_params: Additional parameters for inference
            
        Returns:
            List of model responses
        """
        results = []
        total_inputs = len(batch_inputs)
        
        # More aggressive rate limiting for batch processing
        batch_delay = max(self.request_delay * 2, 2.0)  # Minimum 2 seconds between requests
        
        for i, messages in enumerate(batch_inputs):
            try:
                logging.info(f"Processing batch item {i + 1}/{total_inputs}")
                result = self.run_inference(model_name, messages, inference_params=inference_params)
                results.append(result)
                
                # Add delay between batch requests
                if i < total_inputs - 1:  # Don't wait after the last request
                    logging.info(f"Waiting {batch_delay:.1f} seconds before next request...")
                    time.sleep(batch_delay)
                    
            except (ClientError, ReadTimeoutError, Urllib3ReadTimeoutError) as e:
                if isinstance(e, ClientError):
                    error_code = e.response.get('Error', {}).get('Code', '')
                    error_msg = f"AWS API error ({error_code}): {str(e)}"
                else:
                    error_code = "TimeoutError"
                    error_msg = f"Timeout error: {str(e)}"
                    
                logging.error(f"Batch processing error for input {i + 1}: {error_msg}")
                results.append({"error": error_msg, "error_code": error_code})
                
                # Add extra delay after an error
                if i < total_inputs - 1:
                    error_delay = batch_delay * 2
                    logging.info(f"Error occurred, waiting {error_delay:.1f} seconds before next request...")
                    time.sleep(error_delay)
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logging.error(f"Batch processing error for input {i + 1}: {error_msg}")
                results.append({"error": error_msg})
                
                # Add extra delay after an unexpected error
                if i < total_inputs - 1:
                    error_delay = batch_delay * 2
                    logging.info(f"Unexpected error occurred, waiting {error_delay:.1f} seconds before next request...")
                    time.sleep(error_delay)
        
        return results

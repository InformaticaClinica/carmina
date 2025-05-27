"""
AWS Cloud Provider for LLM operations.

This module implements the AWS-specific provider for accessing LLM models
through AWS services like Amazon Bedrock.
"""

import os
import json
import boto3
from typing import Dict, Any, Optional, List
import logging
from botocore.exceptions import ClientError

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
        
        # LLama models
        "llama-3.3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
        "llama-3.1-70b": "us.meta.llama3-1-70b-instruct-v1:0",
        
        # Other models
        "mistral-7b": "",
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
        
        # Initialize clients
        self.session = self._create_session()
        self.bedrock_runtime = self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )
        # Service flags
        self._is_initialized = False
    
    def connect(self) -> str:
        """
        Establish a connection to AWS services.
        
        Returns:
            Connection status message
        """
        try:
            # Attempt to list available models as a test of connectivity
            self.session.client(
                service_name="bedrock",
                region_name=self.region
            ).list_foundation_models()
            self._is_initialized = True
            return "Connected to AWS API"
        except ClientError as e:
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
                # Determinar si es Claude 3.7 o superior
                is_claude_3_7_plus = "claude-3-7" in model_id.lower() or "claude-3.7" in model_id.lower()
                
                messages = messages.get("messages", [])
                
                # Format for Claude models
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": int(inference_params.get("max_tokens")),
                    "temperature": float(inference_params.get("temperature")),
                }
                
                # Diferentes formatos para diferentes versiones de Claude
                # All Claude models in AWS Bedrock use system as a top-level parameter
                system_messages = [msg for msg in messages if msg.get("role") == "system"]
                user_messages = [msg for msg in messages if msg.get("role") != "system"]
                
                if system_messages:
                    request_body["system"] = system_messages[0].get("content", "")
                
                request_body["messages"] = user_messages
                    
            elif "meta.llama" in model_id:
                # Format for Llama models
                messages_list = messages.get("messages", [])
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
                    "max_gen_len": inference_params.get("max_tokens", 2048),
                    "temperature": inference_params.get("temperature", 0.6),
                    "top_p": inference_params.get("top_p", 0.9),
                }
            elif "mistral" in model_id:
                # Format for Mistral models
                request_body = {
                    "prompt": messages.get("prompt", ""),
                    "max_tokens": inference_params.get("max_tokens"),
                    "temperature": inference_params.get("temperature"),
                }
            else:
                raise ValueError(f"Unsupported model format: {model_id}")
            logging.debug(f"Request body for {model_id}: {json.dumps(request_body)}")
            # Make the actual API call
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse and return the response
            response_body = json.loads(response.get('body').read())
            return response_body
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ValidationException':
                logging.error(f"Request validation error: {str(e)}")
                logging.error(f"Request body was: {json.dumps(request_body)}")
            raise
    
    def batch_process(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, Any]],
        inference_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch mode.
        
        Args:
            model_name: Name of the model to use
            batch_inputs: List of input data dictionaries
            inference_params: Additional parameters for inference
            
        Returns:
            List of model responses
        """
        results = []
        for messages in batch_inputs:
            try:
                result = self.run_inference(model_name, messages, inference_params)
                results.append(result)
            except Exception as e:
                # Log error and continue with other items
                logging.error(f"Batch processing error for input: {e}")
                results.append({"error": str(e)})
        
        return results

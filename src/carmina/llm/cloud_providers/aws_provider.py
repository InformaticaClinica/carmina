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
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude-3.7-sonnet": "anthropic.claude-3-7-sonnet-20240711-v1:0",
        
        # LLama models
        "llama-3-8b": "meta.llama3-8b-instruct-v1:0",
        "llama-3-70b": "meta.llama3-70b-instruct-v1:0",
        
        # Other models
        "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the AWS provider with appropriate credentials and configuration.
        
        Args:
            **kwargs: Additional AWS-specific configuration parameters
        """
        self.region = os.environ.get("AWS_REGION") or kwargs.get("region", "us-east-1")
        self.profile = kwargs.get("profile", None)
        
        # Initialize clients
        self.session = self._create_session()
        self.bedrock_runtime = self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )
        
        # Service flags
        self._is_initialized = False
        
    def _create_session(self) -> boto3.Session:
        """
        Create a properly configured boto3 session.
        
        Returns:
            Configured boto3 Session object
        """
        if self.profile:
            return boto3.Session(profile_name=self.profile, region_name=self.region)
        return boto3.Session(region_name=self.region)
    
    def initialize_environment(self):
        """
        Configure the AWS environment with necessary credentials and permissions.
        
        This method ensures that AWS environment variables are properly set up
        and that the provider has the necessary access to required services.
        
        Returns:
            True if successfully initialized, False otherwise
        """
        try:
            # Verify credentials
            sts = self.session.client("sts")
            sts.get_caller_identity()
            
            # Verify access to Bedrock (for all models)
            try:
                self.bedrock_runtime.list_foundation_models()
            except (ClientError, AttributeError) as e:
                # Alternative check if list_foundation_models is not available
                # Try to get model info for a known model
                model_id = next(iter(self._bedrock_model_ids.values()))
                self.bedrock_runtime.get_model_invoke_config(modelId=model_id)
            
            self._is_initialized = True
            return True
        except ClientError as e:
            logging.error(f"AWS initialization error: {str(e)}")
            return False
    
    def get_credentials(self) -> Dict[str, str]:
        """
        Retrieve current AWS credentials for the session.
        
        Returns:
            Dictionary containing AWS credential information
        """
        credentials = self.session.get_credentials()
        if credentials:
            return {
                "aws_access_key_id": credentials.access_key,
                "aws_secret_access_key": "**redacted**",  # For security
                "region": self.region,
                "initialized": self._is_initialized
            }
        return {"error": "No credentials available", "initialized": False}
        
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
        model_name: str,
        input_data: Dict[str, Any],
        inference_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute inference using AWS Bedrock.

        Args:
            model_name: Name of the model to use
            input_data: Input data for the model
            inference_params: Additional parameters for inference
            
        Returns:
            Model response as a dictionary
            
        Raises:
            ValueError: If the model or request is invalid
            ClientError: If AWS returns an error
        """
        inference_params = inference_params or {}
        
        try:
            model_id = self.get_bedrock_model_id(model_name)
            
            # Format the request according to the provider's requirements
            if "anthropic" in model_id:
                # Format for Claude models
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": inference_params.get("max_tokens", 1000),
                    "temperature": inference_params.get("temperature", 0.7),
                    "messages": input_data.get("messages", [])
                }
            elif "meta.llama" in model_id:
                # Format for Llama models
                request_body = {
                    "prompt": input_data.get("prompt", ""),
                    "max_gen_len": inference_params.get("max_tokens", 512),
                    "temperature": inference_params.get("temperature", 0.7),
                }
            elif "mistral" in model_id:
                # Format for Mistral models
                request_body = {
                    "prompt": input_data.get("prompt", ""),
                    "max_tokens": inference_params.get("max_tokens", 512),
                    "temperature": inference_params.get("temperature", 0.7),
                }
            else:
                raise ValueError(f"Unsupported model format: {model_id}")
            
            # Make the actual API call
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse and return the response
            response_body = json.loads(response.get('body').read())
            return response_body
            
        except ClientError as e:
            logging.error(f"AWS inference error: {str(e)}")
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
        for input_data in batch_inputs:
            try:
                result = self.run_inference(model_name, input_data, inference_params)
                results.append(result)
            except Exception as e:
                # Log error and continue with other items
                logging.error(f"Batch processing error for input: {e}")
                results.append({"error": str(e)})
        
        return results

import logging
from typing import List, Dict, Any
from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider

class MockOllamaProvider(BaseCloudProvider):
    """
    Mock Ollama provider for testing purposes.
    
    This provider simulates the behavior of the LocalProvider with Ollama
    without requiring an actual Ollama installation.
    """
    
    # Mapping for local models to their Ollama tags (same as LocalProvider)
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
        # Gemma models
        "gemma-3-1b": "gemma3:1b",
        "gemma-3-4b": "gemma3:4b",
        # Qwen models
        "qwen-3-1.7b": "qwen3:1.7b"
    }
    
    def __init__(self, name="mock_ollama", **kwargs):
        """
        Initialize the MockOllamaProvider.
        """
        super().__init__(name=name, **kwargs)
        self.base_url = "http://localhost:11434"  # Default Ollama URL
        self.api_endpoint = f"{self.base_url}/api/chat"
        logging.info(f"MockOllamaProvider initialized with endpoint {self.base_url}")
    
    def connect(self):
        """Mock connection to Ollama."""
        logging.info("Mock connection to Ollama established")
        return "Connected to Mock Ollama version 0.1.0"
    
    def run_inference(self, model_id: str, messages: List[Dict[str, str]], **kwargs):
        """
        Mock inference for Ollama models.
        
        Args:
            model_id (str): The model ID to use (e.g., "llama-3.3-70b").
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
            **kwargs: Additional parameters like temperature, max_tokens, etc.
        
        Returns:
            str: A mock response based on the input.
        """
        logging.info(f"Mock inference on local model {model_id}")
        
        # Get the Ollama model name
        ollama_model = self.get_model_id(model_id)
        
        # Extract inference parameters
        inference_params = kwargs.get('inference_params', {})
        
        # Generate mock response based on the last user message
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        
        # Mock responses for common tasks
        if "identifica" in user_message.lower() or "identify" in user_message.lower():
            # Mock anonymization response
            mock_response = self._generate_mock_anonymization_response(user_message)
        elif "test" in user_message.lower():
            mock_response = f"Mock response from {ollama_model}: test completed successfully"
        else:
            mock_response = f"Mock response from {ollama_model}: {user_message[:50]}..."
        
        logging.debug(f"Mock Ollama response: {mock_response}")
        return mock_response
    
    def _generate_mock_anonymization_response(self, text: str) -> str:
        """Generate a mock anonymization response."""
        # Simple mock anonymization - replace common patterns
        anonymized = text
        anonymized = anonymized.replace("Juan", "[NOMBRE]")
        anonymized = anonymized.replace("María", "[NOMBRE]")
        anonymized = anonymized.replace("12345678", "[DNI]")
        anonymized = anonymized.replace("91234567", "[TELEFONO]")
        anonymized = anonymized.replace("Hospital General", "[HOSPITAL]")
        anonymized = anonymized.replace("Dr. García", "[MEDICO]")
        anonymized = anonymized.replace("Madrid", "[CIUDAD]")
        
        return anonymized
    
    def get_model_id(self, model_name: str) -> str:
        """
        Returns the Ollama model name for the given model ID.
        """
        return self._local_model_ids.get(model_name, model_name)
    
    def get_name(self) -> str:
        """
        Returns the name of the cloud provider.
        """
        return "mock_ollama"

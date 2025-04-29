"""
Base LLM strategy interface.

This module defines the abstract base class that all LLM strategies must implement.
It provides a common interface for working with different language models
regardless of their underlying implementation details.
"""
import os

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.carmina.llm.cloud_providers.base_provider import BaseCloudProvider

class BaseLLMStrategy(ABC):
    """
    Abstract base class for LLM strategies.
    
    Defines the interface that all specific LLM implementations must follow.
    """
    
    def __init__(self, model_name: str, cloud_provider: BaseCloudProvider, **kwargs):
        """
        Initialize the LLM strategy.
        
        Args:
            model_name: Name of the specific model to use
            cloud_provider: Cloud provider instance for execution environment
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.cloud_provider = cloud_provider
        self.config = kwargs # TODO: Trace and erase
        self.temperature = float(kwargs.get("temperature", os.getenv("TEMPERATURE", 1.0)))
        self.max_tokens = int(kwargs.get("max_tokens", os.getenv("MAX_TOKENS", 1000)))
        self.top_p = float(kwargs.get("top_p", os.getenv("LLM_TOP_P", 1.0)))
        self.top_k = float(kwargs.get("top_k", os.getenv("LLM_TOP_K", 0)))

    
    @abstractmethod
    def identify(self, text: str, **kwargs) -> str:
        """
        Identify sensitive information in the input text.

        Args:
            text: Input text to process
            **kwargs: Additional identification parameters

        Returns:
            Text with identified sensitive information
        """
        pass

    @abstractmethod
    def batch_identify(self, texts: List[str], **kwargs) -> List[str]:
        """
        Process multiple inputs in batch mode to identify sensitive information.

        Args:
            texts: List of input texts to process
            **kwargs: Additional identification parameters

        Returns:
            List of texts with identified sensitive information
        """
        pass
    
    @abstractmethod
    def get_context_window(self) -> int:
        """
        Get the maximum context window size for this model.
        
        Returns:
            Maximum number of tokens the model can process
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Number of tokens in the text
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this model.
        
        Returns:
            Model name
        """
        return self.model_name
    
    def get_provider(self) -> str:
        """
        Get the cloud provider name.
        
        Returns:
            Cloud provider name
        """
        return self.cloud_provider.get_name()
    
    @abstractmethod
    def process_for_anonymization(self, text: str, strategy: str) -> Dict[str, Any]:
        """
        Process text specifically for anonymization tasks.
        
        Args:
            text: Text to anonymize
            strategy: Anonymization strategy to apply (e.g., 'label', 'substitute')
            
        Returns:
            Dictionary with processed results including the anonymized text
            and any identified entities
        """
        pass

    @abstractmethod
    def process_for_anonymization(self, text: str, mode: str) -> Dict[str, Any]:
        """
        Process text specifically for anonymization tasks.
        
        Args:
            text: Text to anonymize
            mode: Anonymization mode to apply (e.g., 'label', 'substitute')
            
        Returns:
            Dictionary with processed results including the anonymized text
            and any identified entities
        """
        pass
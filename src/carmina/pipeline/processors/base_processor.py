"""
Base processor interface for pipeline components.

This module defines the abstract base class that all pipeline processors must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy

class BaseProcessor(ABC):
    """
    Abstract base class for all pipeline processors.
    
    Defines the common interface for processors in the anonymization pipeline.
    """
    
    def __init__(self, llm_strategy: BaseLLMStrategy):
        """
        Initialize the processor with an LLM strategy.
        
        Args:
            llm_strategy: The LLM strategy to use for processing
        """
        self.llm_strategy = llm_strategy
        
    @abstractmethod
    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process the input text according to this processor's function.
        
        Args:
            text: The input text to process
            **kwargs: Additional processor-specific parameters
            
        Returns:
            Dictionary containing the processing results
        """
        pass
        
    def _validate_input(self, text: str) -> bool:
        """
        Validate that the input text meets the requirements for this processor.
        
        Args:
            text: The input text to validate
            
        Returns:
            True if valid, False otherwise
        """
        return text and isinstance(text, str)
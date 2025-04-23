"""
Entity identification processor.

This processor identifies sensitive information in text using the LLM.
"""

import logging
from typing import Dict, Any, List
from src.carmina.pipeline.processors.base_processor import BaseProcessor

class IdentificationProcessor(BaseProcessor):
    """
    Processor that identifies sensitive information in text.
    
    This processor uses the LLM strategy to detect entities that require anonymization.
    """
    
    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process the input text to identify sensitive entities.
        
        Args:
            text: The input text to process
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing identified entities
        """
        if not self._validate_input(text):
            return {"entities": {}, "error": "Invalid input text"}
            
        try:
            # Call the LLM strategy to extract entities
            entities = self.llm_strategy.identify(text)
            
            return {
                "entities": entities,
                "count": len(entities)
            }
            
        except Exception as e:
            logging.error(f"Error in identification process: {e}")
            return {"entities": {}, "error": str(e)}
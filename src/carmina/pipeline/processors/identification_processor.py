"""
Entity identification processor.

This processor identifies sensitive information in text using the LLM.
"""

import logging
from typing import Dict, Any, List
from src.carmina.pipeline.processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


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
            **kwargs: Additional parameters including 'filename' for logging

        Returns:
            Dictionary containing identified entities
        """
        if not self._validate_input(text):
            return {"entities": {}, "error": "Invalid input text"}

        try:
            text_identify = self.llm_strategy.identify(text)
            entities = self._get_brackets_entities(text_identify)
            logging.info(f"Identified entities: {entities}")
            return {
                "anonymized_text": text_identify,
                "entities": entities,
            }
        except Exception as e:
            logging.error(f"Error in identification process: {e}")
            return {"entities": {}, "error": str(e)}

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Divide el texto en chunks de aproximadamente chunk_size tokens.
        Intenta mantener límites de palabras para coherencia.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = self.llm_strategy.count_tokens(word + " ")
            if current_tokens + word_tokens > chunk_size and current_chunk:
                # Unir el chunk actual y empezar uno nuevo
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        # Agregar el último chunk si existe
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

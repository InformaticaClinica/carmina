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
        filename = kwargs.get("filename", "unknown")

        if not self._validate_input(text):
            return {"entities": {}, "error": "Invalid input text"}

        try:
            # Always chunk text into 100-token chunks
            chunk_size = 100
            chunks = self._chunk_text(text, chunk_size)
            identified_chunks = []
            chunks_info = []

            for i, chunk in enumerate(chunks, 1):
                logging.info(f"Processing chunk {i}/{len(chunks)} for file {filename}")
                identified_chunk = self.llm_strategy.identify(chunk)
                identified_chunks.append(identified_chunk)

                # Extract entities for this chunk
                entities_chunk = self._get_brackets_entities(identified_chunk)
                chunks_info.append(
                    {
                        "chunk_text": chunk,
                        "anonymized_text": identified_chunk,
                        "entities": entities_chunk,
                    }
                )

            # Unir los resultados
            text_identify = " ".join(identified_chunks)

            entities = self._get_brackets_entities(text_identify)
            logging.info(f"Identified entities: {entities}")
            return {
                "anonymized_text": text_identify,
                "entities": entities,
                "chunks": chunks_info,
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

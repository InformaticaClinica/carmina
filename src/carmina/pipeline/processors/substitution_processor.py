"""
Entity substitution processor.

This processor anonymizes text by replacing sensitive entities with realistic alternatives.
"""

import logging
from typing import Dict, Any, List
from src.carmina.pipeline.processors.base_processor import BaseProcessor

class SubstitutionProcessor(BaseProcessor):
    """
    Processor that substitutes sensitive information in text.
    
    This processor replaces sensitive entities with realistic alternatives.
    """
    
    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process the text to replace sensitive entities with substitutes.

        Args:
            text: The input text to anonymize
            **kwargs: Additional parameters including pre-identified entities and 'filename' for logging

        Returns:
            Dictionary containing the anonymized text and metadata
        """
        filename = kwargs.get("filename", "unknown")

        if not self._validate_input(text):
            return {"anonymized_text": "", "error": "Invalid input text"}

        # Extract entities from kwargs if available
        entities = kwargs.get("entities", {})

        try:
            # Always chunk text into 100-token chunks
            chunk_size = 100
            chunks = self._chunk_text(text, chunk_size)
            substituted_chunks = []

            for i, chunk in enumerate(chunks, 1):
                logging.info(f"Processing chunk {i}/{len(chunks)} for file {filename}")
                substituted_chunk = self.llm_strategy.process_for_anonymization(chunk, "substitute")
                substituted_chunks.append(substituted_chunk)

            # Unir los resultados
            result = "".join(substituted_chunks)

            # Extract labels for evaluation
            # labels = self._extract_substitution_labels(result)

            return {
                "anonymized_text": result,
            }

        except Exception as e:
            logging.error(f"Error in substitution process: {e}")
            return {"anonymized_text": text, "entities": entities, "error": str(e)}

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
            
    def _extract_substitution_labels(self, substitution_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Convert substitution map to label format for evaluation.
        
        Args:
            substitution_map: Map of original values to substituted values
            
        Returns:
            List of labels in standard format for evaluation
        """
        labels = []
        # Convert substitution map to standard label format for evaluation
        return labels
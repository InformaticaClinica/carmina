"""
Entity labeling processor.

This processor anonymizes text by replacing sensitive entities with labels.
"""

import logging
from typing import Dict, Any, List
from src.carmina.pipeline.processors.base_processor import BaseProcessor

class LabelingProcessor(BaseProcessor):
    """
    Processor that labels sensitive information in text.
    
    This processor replaces sensitive entities with standardized labels.
    """
    
    def process(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process the text to replace sensitive entities with labels.

        Args:
            text: The input text to anonymize
            **kwargs: Additional parameters including pre-identified entities and 'filename' for logging

        Returns:
            Dictionary containing the labeled text and metadata
        """
        filename = kwargs.get("filename", "unknown")

        if not self._validate_input(text):
            return {"anonymized_text": "", "error": "Invalid input text"}

        try:
            # Verificar si necesita chunking
            total_tokens = self.llm_strategy.count_tokens(text)
            context_window = self.llm_strategy.get_context_window()

            if total_tokens > context_window - 500:  # Buffer de seguridad
                # Implementar chunking en chunks de 100 tokens
                chunks = self._chunk_text(text, 100)
                labeled_chunks = []

                for i, chunk in enumerate(chunks, 1):
                    logging.info(f"Processing chunk {i}/{len(chunks)} for file {filename}")
                    labeled_chunk = self.llm_strategy.process_for_anonymization(chunk, "label")
                    labeled_chunks.append(labeled_chunk)

                # Unir los resultados
                result = "".join(labeled_chunks)
            else:
                result = self.llm_strategy.process_for_anonymization(text, "label")

            # Extract labels for evaluation
            labels = self._get_brackets_entities(result)

            return {
                "anonymized_text": result,
                "labels": labels
            }

        except Exception as e:
            logging.error(f"Error in labeling process: {e}")

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
            
    def _extract_labels(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract labels from the anonymized text.
        
        Args:
            text: Anonymized text with labels
            
        Returns:
            List of labels with their positions and types
        """
        # Implementation would parse labels like [**NAME**] in the text
        # This is a simplified placeholder
        labels = []
        # Logic to extract and return labels from text
        return labels
    
    
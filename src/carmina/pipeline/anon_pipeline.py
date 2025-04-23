"""
Anonymization pipeline orchestrator.

This module defines the main pipeline that orchestrates the entire
anonymization process from raw text to fully anonymized output.
"""

import logging
from typing import Dict, List, Any, Optional
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.pipeline.processors.base_processor import BaseProcessor
from src.carmina.pipeline.processors.identification_processor import IdentificationProcessor
from src.carmina.pipeline.processors.labeling_processor import LabelingProcessor
from src.carmina.pipeline.processors.substitution_processor import SubstitutionProcessor

class AnonymizationPipeline:
    """
    Main pipeline for text anonymization processes.
    
    This pipeline orchestrates the complete anonymization flow, including:
    - Entity identification
    - Entity labeling or substitution
    """
    
    def __init__(self, llm_strategy: BaseLLMStrategy):
        """
        Initialize the anonymization pipeline with the specified LLM strategy.
        
        Args:
            llm_strategy: The LLM strategy to use for processing
        """
        self.llm_strategy = llm_strategy
        self.anonymization_mode = llm_strategy.anonymization_mode
        
        # Configure processors based on mode
        self.identification = IdentificationProcessor(llm_strategy)
        
        # Initialize the right processor based on anonymization mode
        if self.anonymization_mode == "label":
            self.anonymizer = LabelingProcessor(llm_strategy)
        elif self.anonymization_mode == "substitute":
            self.anonymizer = SubstitutionProcessor(llm_strategy)
        else:
            raise ValueError(f"Unsupported anonymization mode: {self.anonymization_mode}")
        
        logging.info(f"Pipeline initialized with {llm_strategy.get_name()} using {self.anonymization_mode} mode")
        
    def run(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the complete anonymization pipeline on a list of records.
        
        Args:
            records: List of data records to process
                Each record should be a dictionary with at least a 'text' field
                
        Returns:
            List of processed records with anonymized text and metadata
        """
        results = []
        
        for record in records:
            try:
                # Step 1: Extract the text content to process
                text = record.get('text', '')
                if not text:
                    logging.warning(f"Empty text found in record, skipping")
                    results.append({**record, "error": "Empty text"})
                    continue
                
                # Step 2: Run identification to find sensitive entities
                identified_result = self.identification.process(text)
                
                # Step 3: Run anonymization (labeling or substitution)
                anonymized_result = self.anonymizer.process(
                    text, 
                    entities=identified_result.get('entities', {})
                )
                
                # Step 4: Combine all results into the output
                output = {
                    **record,
                    "anonymized_text": anonymized_result.get('anonymized_text', ''),
                    "entities": anonymized_result.get('entities', {}),
                    "predicted_labels": anonymized_result.get('labels', []),
                }
                
                results.append(output)
                
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                results.append({**record, "error": str(e)})
        
        return results
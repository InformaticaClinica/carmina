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
        if self.anonymization_mode == "identify":
            self.anonymizer = None
        elif self.anonymization_mode == "label":
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
                
                # Step 2: Run identification to find sensitive entitiesç
                identified_result = self.identify(text)
                   
                # Step 3: Run anonymization (labeling or substitution)
                anonymized_result = self.anonymize(text=text.get("anonymized_text"), identified_result=identified_result.get("entities"))
                
                # Step 4: Combine all results into the output
                output = {
                    **record,
                    "identified_text": identified_result.get('anonymized_text', ''),
                    "entities_identified": identified_result.get('entities', {}),
                    "anonymized_text": anonymized_result.get('anonymized_text', ''),
                    "entities_anonymized": anonymized_result.get('entities', {}),
                    "predicted_labels": anonymized_result.get('labels', []),
                }
                results.append(output)
                
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                results.append({**record, "error": str(e)})
        
        return results

    def identify(self, text: str) -> Dict[str, Any]:
        """
        Identify sensitive entities in the input text.
        
        Args:
            text: The input text to identify entities in
            
        Returns:
            Dictionary with identified entities and their labels
        """
        if self.identification is not None:
            identified_result = self.identification.process(text)
            return identified_result
        else:
            return {}

    def anonymize(self, text: str, identified_result:Dict[List, Any]) -> str:
        """
        Anonymize the input text using the configured LLM strategy.
        
        Args:
            text: The input text to anonymize
            
        Returns:
            Anonymized text
        """
        if self.anonymizer is not None:
            return self.anonymizer.process(text)
        else:
            return identified_result
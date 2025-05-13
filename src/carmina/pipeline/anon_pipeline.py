"""
Anonymization pipeline orchestrator.

This module defines the main pipeline that orchestrates the entire
anonymization process from raw text to fully anonymized output.
"""

import logging
logger = logging.getLogger(__name__)

import os
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
        
        # Configure document processing limit from environment variable
        self.max_documents = self._get_max_documents_from_env()
        self.processed_count = 0
        
        logging.info(f"Pipeline initialized with {llm_strategy.get_name()} using {self.anonymization_mode} mode")
        logging.info(f"Maximum documents to process: {self.max_documents if self.max_documents else 'No limit'}")
    
    def _get_max_documents_from_env(self) -> Optional[int]:
        """
        Get the maximum number of documents to process from environment variables.
        
        Returns:
            Optional[int]: The maximum number of documents to process or None if no limit
        """
        max_docs_str = os.environ.get("MAX_DOCUMENTS_TO_PROCESS")
        if not max_docs_str:
            return None
            
        try:
            max_docs = int(max_docs_str)
            return max_docs if max_docs > 0 else None
        except ValueError:
            logging.warning("Invalid MAX_DOCUMENTS_TO_PROCESS value in .env, should be a positive integer")
            return None
        
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
            # Check if we've reached the processing limit
            if self.max_documents is not None and self.processed_count >= self.max_documents:
                logging.info(f"Reached maximum document limit ({self.max_documents}), stopping processing")
                break
                
            try:
                # Step 1: Extract the text content to process
                text = record.get('text', '')
                if not text:
                    logging.warning(f"Empty text found in record, skipping")
                    results.append({**record, "error": "Empty text"})
                    continue
                
                # Step 2: Run identification to find sensitive entities
                identified_result = self.identify(text)

                # Step 3: Run anonymization (labeling or substitution)
                anonymized_result = self.anonymize(text=identified_result.get("anonymized_text"), identified_result=identified_result.get("entities"))

                # Step 4: Combine all results into the output
                output = {
                    **record,
                    "gt_raw_entities": self.identification._get_brackets_entities(record.get('identify', '')),
                    "gt_masked_entities": self.identification._get_brackets_entities(record.get('masked_text', '')),
                    "identified_text": identified_result.get('anonymized_text', ''),
                    "entities_identified": identified_result.get('entities', {}),
                    "anonymized_text": anonymized_result.get('anonymized_text', ''),
                    "entities_anonymized": anonymized_result.get('labels', {}),
                }
                results.append(output)
                self.processed_count += 1
                
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                results.append({**record, "error": str(e)})
        
        logging.info(f"Completed processing {self.processed_count} documents")
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

    def anonymize(self, text: str, identified_result:Dict[List, Any]) -> Dict[str, Any]:
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
"""
Anonymization pipeline orchestrator.

This module defines the main pipeline that orchestrates the entire
anonymization process from raw text to fully anonymized output.
"""

import nltk
import logging
import os
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize
from typing import Dict, List, Any, Optional
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.pipeline.processors.base_processor import BaseProcessor
from src.carmina.pipeline.processors.identification_processor import (
    IdentificationProcessor,
)
from src.carmina.pipeline.processors.labeling_processor import LabelingProcessor
from src.carmina.pipeline.processors.substitution_processor import SubstitutionProcessor

MAX_CHUNK_SIZE = 100
CHUNK_BOOL = False


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

        # Configure document processing limit from environment variable
        self.first_document = self._get_first_document()
        self.max_documents = self._get_max_documents_from_env()
        self.max_workers = self._get_max_workers_from_env()
        self.processed_count = 0
        self.results_lock = threading.Lock()  # Thread-safe results access
        
        # Setup incremental save path
        debug_dir = os.getenv("DEBUG_DIR", "data/outputs/debug/")
        os.makedirs(debug_dir, exist_ok=True)
        self.incremental_save_path = os.path.join(debug_dir, f"output_{llm_strategy.get_name()}_partial.json")
        
        # Handle existing partial file from previous run
        if os.path.exists(self.incremental_save_path):
            # Rename previous partial as backup
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.incremental_save_path.replace("_partial.json", f"_partial_backup_{timestamp}.json")
            os.rename(self.incremental_save_path, backup_path)
            logging.info(f"Previous partial results backed up to: {backup_path}")

        logging.info(
            f"Pipeline initialized with {llm_strategy.get_name()} using {self.anonymization_mode} mode"
        )
        logging.info(
            f"Maximum documents to process: {self.max_documents if self.max_documents else 'No limit'}"
        )
        logging.info(f"Parallel workers: {self.max_workers}")
        logging.info(f"Incremental results will be saved to: {self.incremental_save_path}")

    def _get_first_document(self) -> Optional[int]:
        return int(os.environ.get("FIRST_DOCUMENT_TO_PROCESS", "0"))

    def _get_max_workers_from_env(self) -> int:
        """
        Get the maximum number of parallel workers from environment variables.

        Returns:
            int: The maximum number of workers (default: 1 for sequential processing)
        """
        max_workers_str = os.environ.get("MAX_WORKERS")
        if not max_workers_str:
            return 1  # Default: sequential processing

        try:
            max_workers = int(max_workers_str)
            return max(1, max_workers)  # Ensure at least 1 worker
        except ValueError:
            logging.warning(
                "Invalid MAX_WORKERS value in .env, should be a positive integer. Using default: 1"
            )
            return 1
    
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
            logging.warning(
                "Invalid MAX_DOCUMENTS_TO_PROCESS value in .env, should be a positive integer"
            )
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
        # Filter records based on first_document limit
        records_to_process = []
        for idx, record in enumerate(records):
            if self.first_document is not None and (idx + 1) < self.first_document:
                logging.info(f"Skipping record {idx + 1} (first document limit not reached)")
                continue
            if self.max_documents is not None and len(records_to_process) >= self.max_documents:
                break
            records_to_process.append(record)
        
        total_to_process = len(records_to_process)
        logging.info(f"Processing {total_to_process} documents with {self.max_workers} parallel worker(s)")
        
        # Process documents in parallel or sequentially
        if self.max_workers == 1:
            # Sequential processing (original behavior)
            results = self._process_sequential(records_to_process, total_to_process)
        else:
            # Parallel processing with ThreadPoolExecutor
            results = self._process_parallel(records_to_process, total_to_process)
        
        logging.info(f"Completed processing {self.processed_count} documents")
        return results
    
    def _process_sequential(self, records: List[Dict[str, Any]], total_to_process: int) -> List[Dict[str, Any]]:
        """
        Process records sequentially (original behavior).
        
        Args:
            records: List of records to process
            total_to_process: Total number of documents to process
            
        Returns:
            List of processed records
        """
        results = []
        for record in records:
            result = self._process_single_record(record, total_to_process)
            results.append(result)
            # Save partial results after each document
            with self.results_lock:
                self._save_partial_results(results)
        return results
    
    def _process_parallel(self, records: List[Dict[str, Any]], total_to_process: int) -> List[Dict[str, Any]]:
        """
        Process records in parallel using ThreadPoolExecutor.
        
        Args:
            records: List of records to process
            total_to_process: Total number of documents to process
            
        Returns:
            List of processed records (order preserved)
        """
        results = [None] * len(records)  # Pre-allocate to preserve order
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with their index
            future_to_index = {
                executor.submit(self._process_single_record, record, total_to_process): idx
                for idx, record in enumerate(records)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                    
                    # Save partial results (thread-safe)
                    with self.results_lock:
                        # Get only completed results (non-None) for saving
                        completed_results = [r for r in results if r is not None]
                        self._save_partial_results(completed_results)
                        
                except Exception as e:
                    logging.error(f"Error in parallel processing: {e}")
                    results[idx] = {**records[idx], "error": str(e)}
        
        return results
    
    def _process_single_record(self, record: Dict[str, Any], total_to_process: int) -> Dict[str, Any]:
        """
        Process a single record through the anonymization pipeline.
        
        Args:
            record: Record to process
            total_to_process: Total number of documents being processed
            
        Returns:
            Processed record with anonymization results
        """
        # Generate unique ID for this processing task
        task_id = str(uuid.uuid4())[:8]
        
        try:
            # Step 1: Extract the text content to process
            text = record.get("text", "")
            filename = record.get("id", "unknown")
            
            if not text:
                logging.warning(f"[{task_id}] Empty text found in record {filename}, skipping")
                return {**record, "error": "Empty text"}
            
            logging.info(f"[{task_id}] Processing {filename}")
            
            # Step 2: Anonymize
            if CHUNK_BOOL:
                chunk_text = self.get_text_chunks(text)
                processed_chunks = self.run_chunk_identify(chunk_text)
            else:
                # TODO: Refactor
                identified = self.identify(text)
                processed_chunks = {
                    "original_text": text,
                    "identified_text": identified.get("anonymized_text"),
                    "entities_identified": identified.get("entities"),
                }
                if self.anonymization_mode != "identify":
                    anonymized = self.anonymize(
                        text=identified.get("anonymized_text")
                    )
                    processed_chunks["anonymized_text"] = anonymized.get(
                        "anonymized_text"
                    )
                    processed_chunks["entities_anonymized"] = anonymized.get(
                        "entities"
                    )
            
            # Step 3: Prepare output
            output = {
                **record,  # Preserve original record data (ground truth)
                "namefile": filename,
                "chunks": processed_chunks,
                "processing_task_id": task_id,  # Add unique identifier
            }
            
            # Thread-safe increment of processed count
            with self.results_lock:
                self.processed_count += 1
                current_count = self.processed_count
            
            logging.info(f"[{task_id}] {filename} completed ({current_count} / {total_to_process})")
            return output
            
        except Exception as e:
            logging.error(f"[{task_id}] Error processing record {record.get('id', 'unknown')}: {e}")
            return {**record, "error": str(e), "processing_task_id": task_id}
    
    def _save_partial_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Save partial results incrementally after each document.
        
        Args:
            results: Current list of processed results
        """
        try:
            with open(self.incremental_save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.debug(f"Partial results saved: {len(results)} documents")
        except Exception as e:
            logging.error(f"Error saving partial results: {e}")

    def run_chunk_identify(self, chunks):
        """
        Proceess chunks identify and anonymized

        Args:
            text: The input is a list of chunks

        Returns:
            processed_chunks: The output is a list of dictionaries of each chunk
        """
        processed_chunks = []
        logging.info(f"Number of chunks {len(chunks)}")
        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            logging.info(f"Iteration chunk {i + 1}/{n_chunks}")
            identified = self.identify(chunk)
            processed_chunk = {
                "original_text": chunk,
                "identified_text": identified.get("anonymized_text"),
                "entities_identified": identified.get("entities"),
            }
            if self.anonymization_mode != "identify":
                anonymized = self.anonymize(text=identified.get("anonymized_text"))
                processed_chunk["anonymized_text"] = anonymized.get("anonymized_text")
                processed_chunk["entities_anonymized"] = anonymized.get("entities")
            processed_chunks.append(processed_chunk)
        return processed_chunks

    def identify(self, text: str) -> Dict[str, Any]:
        """
        Identify sensitive entities in the input text.

        Args:
            text: The input text to identify entities in
            filename: The filename for logging purposes

        Returns:
            Dictionary with identified entities and their labels
        """
        if self.identification is not None:
            identified_result = self.identification.process(text)
            return identified_result
        else:
            return {}

    def anonymize(self, text: str) -> Dict[str, Any]:
        """
        Anonymize the input text using the configured LLM strategy.

        Args:
            text: The input text to anonymize
            identified_result: The result from identification
            filename: The filename for logging purposes

        Returns:
            Anonymized text
        """
        if self.anonymizer is not None:
            return self.anonymizer.process(text)

    # TODO: Refactor file
    @staticmethod
    def get_text_chunks(text: str) -> List[str]:
        """Splits text into chunks of a maximum size, respecting sentence boundaries."""
        sentences = sent_tokenize(text, language="spanish")

        text_chunks = []
        current_chunk_words = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            if (
                current_word_count + sentence_word_count > MAX_CHUNK_SIZE
                and current_chunk_words
            ):
                text_chunks.append(" ".join(current_chunk_words))
                current_chunk_words = sentence_words
                current_word_count = sentence_word_count
            else:
                current_chunk_words.extend(sentence_words)
                current_word_count += sentence_word_count

        if current_chunk_words:
            text_chunks.append(" ".join(current_chunk_words))

        return text_chunks

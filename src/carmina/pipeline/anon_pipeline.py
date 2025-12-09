"""
Anonymization pipeline orchestrator.

This module defines the main pipeline that orchestrates the entire
anonymization process from raw text to fully anonymized output.
"""

import logging

logger = logging.getLogger(__name__)

import nltk

nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


import os
from typing import Dict, List, Any, Optional
from src.carmina.llm.strategies.base_strategy import BaseLLMStrategy
from src.carmina.pipeline.processors.base_processor import BaseProcessor
from src.carmina.pipeline.processors.identification_processor import (
    IdentificationProcessor,
)
from src.carmina.pipeline.processors.labeling_processor import LabelingProcessor
from src.carmina.pipeline.processors.substitution_processor import SubstitutionProcessor

MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "100"))


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
        self.processed_count = 0

        logging.info(
            f"Pipeline initialized with {llm_strategy.get_name()} using {self.anonymization_mode} mode"
        )
        logging.info(
            f"Maximum documents to process: {self.max_documents if self.max_documents else 'No limit'}"
        )

    def _get_first_document(self) -> Optional[int]:
        return int(os.environ.get("FIRST_DOCUMENT_TO_PROCESS"))

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
        results = []
        total_to_process = (
            self.max_documents
            if self.max_documents
            else len(records) - (self.first_document - 1 if self.first_document else 0)
        )
        count = 0
        for record in records:
            # Check if we've reached the processing limit
            if self.first_document is not None and (count + 1) < self.first_document:
                logging.info(
                    f"Skipping record {count + 1} (first document limit not reached)"
                )
                count += 1
                continue

            if (
                self.max_documents is not None
                and self.processed_count >= self.max_documents
            ):
                logging.info(
                    f"Reached maximum document limit ({self.max_documents}), stopping processing"
                )
                break

            try:
                # Step 1: Extract the text content to process
                text = record.get("text", "")
                if not text:
                    logging.warning("Empty text found in record, skipping")
                    results.append({**record, "error": "Empty text"})
                    continue

                # Step 2: Anonimized
                chunk_text = self.get_text_chunks(text)
                processed_chunks = self.run_chunk_identify(chunk_text)

                # Step 3: Store
                filename = record.get("id", "unknown")
                output = {
                    **record,  # Preserve original record data (ground truth)
                    "namefile": filename,
                    "chunks": processed_chunks,
                }
                results.append(output)
                self.processed_count += 1
                logging.info(f"{filename} {self.processed_count} / {total_to_process}")
            except Exception as e:
                logging.error(f"Error processing record: {e}")
                results.append({**record, "error": str(e)})

        logging.info(f"Completed processing {self.processed_count} documents")
        return results

    def run_chunk_identify(self, chunks):
        """
        Proceess chunks identify and anonymized

        Args:
            text: The input is a list of chunks

        Returns:
            processed_chunks: The output is a list of dictionaries of each chunk
        """
        processed_chunks = []
        logging.info(f"Number of chunks {chunks}")
        for chunk in chunks:
            identified = self.identify(chunk)
            processed_chunk = {
                "original_text": chunk,
                "identified_text": identified.get("anonymized_text"),
                "entities_identified": identified.get("entities"),
            }
            if self.anonymization_mode != "identify":
                anonymized = self.anonymize(
                    text=identified.get("anonymized_text"), identified_result=identified
                )
                processed_chunk["anonymized_text"] = (
                    anonymized.get("anonymized_text"),
                )
                processed_chunk["entities_anonymized"] = (anonymized.get("entities"),)
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

    def anonymize(
        self, text: str, identified_result: Dict[str, Any], filename: str = "unknown"
    ) -> Dict[str, Any]:
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
            return self.anonymizer.process(text, filename=filename)
        else:
            return identified_result

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
